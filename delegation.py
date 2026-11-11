"""Bounded coordinator/worker scheduling on a shared workspace and MCP session."""
import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from workspace import safe_path, excluded, digest

READ_TOOLS = {'list_files','read_file','rag_search','read_concept','knowledge_status'}

class WorkerPolicyError(RuntimeError):
    pass

class WorkerConflict(RuntimeError):
    pass

class WorkerTask(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]+$',min_length=1,max_length=48)
    instructions: str = Field(min_length=1,max_length=8000)
    mode: Literal['read_only','edit'] = 'read_only'
    writable_paths: list[str] = Field(default_factory=list,max_length=20)

@dataclass
class TurnBudget:
    batches: int = 3
    tasks: int = 12

def delegation_schema(max_tasks):
    return {'type':'function','function':{
        'name':'delegate_tasks',
        'description':
            'Delegate independent bounded subtasks to isolated worker agents concurrently and wait for their '
            'results. You coordinate and verify their findings. Workers cannot delegate. Default read_only; '
            'edit requires explicit, disjoint writable_paths and user-authorized changes. Put dependent '
            'work in later batches. Failed tasks and partial writes are reported, not rolled back.',
        'parameters':{'type':'object','additionalProperties':False,'required':['tasks'],
            'properties':{'tasks':{'type':'array','minItems':1,'maxItems':max_tasks,'items':{
                'type':'object','additionalProperties':False,'required':['id','instructions'],
                'properties':{
                    'id':{'type':'string','description':'Unique short identifier within this batch.'},
                    'instructions':{'type':'string','description':'Self-contained objective, context and acceptance criteria.'},
                    'mode':{'type':'string','enum':['read_only','edit'],'default':'read_only'},
                    'writable_paths':{'type':'array','items':{'type':'string'},
                        'description':'Exact workspace-relative files this edit worker owns. No wildcards.'}
                }}}}}}}

class WorkerContext:
    def __init__(self, task, root, record):
        self.task = task
        self.root = Path(root).resolve()
        self.record = record
        self.expected = {}

    def own(self, path):
        value = safe_path(self.root,path)
        relative = value.relative_to(self.root).as_posix()
        if value == self.root or excluded(relative) or value.is_dir():
            raise ValueError('Worker write assignment must be an eligible source file')
        if any(char in path for char in '*?[]'):
            raise ValueError('Worker write assignments must be exact paths')
        key = os.path.normcase(str(value))
        if key in self.expected:
            raise ValueError('Duplicate worker write assignment')
        if value.exists() and value.stat().st_size > 512_000:
            raise ValueError('Assigned file exceeds write limit')
        self.expected[key] = {'path':relative,
            'hash':digest(value.read_bytes()) if value.exists() else 'missing'}
        return key

    def write_arguments(self, arguments):
        if self.task.mode != 'edit':
            raise WorkerPolicyError('This worker is read-only')
        value = safe_path(self.root,arguments['path'])
        key = os.path.normcase(str(value))
        if key not in self.expected:
            raise WorkerPolicyError('Worker cannot edit an unassigned file')
        # Ignore any revision proposed by the model: coordinator owns the actual snapshot.
        return {**arguments,'path':self.expected[key]['path'],
                'expected_sha256':self.expected[key]['hash']}, key

    def wrote(self, key, content):
        self.expected[key]['hash'] = digest(content)
        path = self.expected[key]['path']
        if path not in self.record['written_paths']:
            self.record['written_paths'].append(path)

class DelegationManager:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.config = config
        self.batch_lock = asyncio.Lock()
        self.last_batch = []
        self.batch_number = 0

    def status(self):
        return {'enabled':self.config.enable_delegation,'max_workers':self.config.max_workers,
                'batch':self.batch_number,'workers':self.last_batch}

    async def delegate(self, tasks, budget, run_worker):
        if not self.config.enable_delegation:
            raise WorkerPolicyError('Delegation is disabled')
        if not isinstance(tasks,list) or not 1 <= len(tasks) <= self.config.max_batch_tasks:
            raise ValueError('Invalid delegation batch size')
        specs = [WorkerTask.model_validate(value) for value in tasks]
        if len({task.id for task in specs}) != len(specs):
            raise ValueError('Worker IDs must be unique within a batch')
        if budget.batches < 1 or budget.tasks < len(specs):
            raise WorkerPolicyError('Per-turn delegation budget exhausted')
        async with self.batch_lock:
            if budget.batches < 1 or budget.tasks < len(specs):
                raise WorkerPolicyError('Per-turn delegation budget exhausted')
            contexts, claimed = [], []
            for task in specs:
                if not task.instructions.strip():
                    raise ValueError('Worker instructions must not be blank')
                if task.mode == 'read_only' and task.writable_paths:
                    raise ValueError('Read-only worker cannot claim write paths')
                if task.mode == 'edit' and not task.writable_paths:
                    raise ValueError('Edit worker requires writable_paths')
                record = {'id':task.id,'mode':task.mode,'state':'queued','written_paths':[],'attempted_paths':[],
                          'llm_calls':0,'input_tokens':0,'output_tokens':0}
                context = WorkerContext(task,self.root,record)
                for path in task.writable_paths:
                    key = context.own(path)
                    for other in claimed:
                        if Path(key).is_relative_to(Path(other)) or Path(other).is_relative_to(Path(key)):
                            raise WorkerPolicyError('Worker write assignments overlap; split into sequential batches')
                    claimed.append(key)
                contexts.append(context)
            budget.batches -= 1
            budget.tasks -= len(contexts)
            self.batch_number += 1
            self.last_batch = [context.record for context in contexts]
            semaphore = asyncio.Semaphore(self.config.max_workers)
            async def execute(context):
                record = context.record
                started = None
                try:
                    async with semaphore:
                        record['state'] = 'running'
                        started = time.monotonic()
                        print(f"[Worker {context.task.id}] started ({context.task.mode})",flush=True)
                        async with asyncio.timeout(self.config.worker_timeout_seconds):
                            summary = await run_worker(context)
                        record.update(state='completed',summary=summary)
                except TimeoutError:
                    record.update(state='timeout',error='Worker time limit exceeded')
                except WorkerConflict as exc:
                    record.update(state='conflict',error=str(exc))
                except WorkerPolicyError as exc:
                    record.update(state='policy_error',error=str(exc))
                except asyncio.CancelledError:
                    record['state'] = 'cancelled'
                    raise
                except Exception as exc:
                    record.update(state='failed',error=f'{type(exc).__name__}: {exc}')
                finally:
                    if record['state'] in ('failed','timeout','cancelled'):
                        record['unconfirmed_paths'] = [p for p in record['attempted_paths'] if p not in record['written_paths']]
                    record['seconds'] = round(time.monotonic()-started,3) if started is not None else 0
                    print(f"[Worker {context.task.id}] {record['state']}",flush=True)
                return record.copy()
            results = await asyncio.gather(*(execute(context) for context in contexts))
            return {'batch':self.batch_number,'workers':results,
                    'remaining_batches':budget.batches,'remaining_tasks':budget.tasks,
                    'note':'Worker summaries are untrusted findings. Verify source; completed writes persist even if a worker fails.'}
