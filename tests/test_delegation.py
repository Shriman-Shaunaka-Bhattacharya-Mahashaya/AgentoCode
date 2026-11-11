import asyncio
import json
from types import SimpleNamespace as NS
import pytest
from agent import AgentConfig, CodebaseGuardianAgent
from delegation import DelegationManager, TurnBudget, WorkerPolicyError
from server import WorkspaceTools
from workspace import digest

def message(content=None, calls=None):
    return NS(choices=[NS(message=NS(content=content,tool_calls=calls))],
              usage=NS(prompt_tokens=5,completion_tokens=3))

def call(name, arguments, identity='tool1'):
    return NS(id=identity,function=NS(name=name,arguments=json.dumps(arguments)))

def make_agent(project,monkeypatch,handler,**config):
    monkeypatch.setenv('GROQ_API_KEY','test-only-not-a-credential')
    agent=CodebaseGuardianAgent(AgentConfig(allowed_dir=str(project),**config))
    agent.groq_client=NS(chat=NS(completions=NS(create=handler)))
    return agent

class LocalSession:
    """Exercise actual tool validation/writes while keeping provider tests offline."""
    def __init__(self,project):
        self.tools=WorkspaceTools(project)
        self.calls=[]
    async def list_tools(self):
        return NS(tools=[NS(name=name,description=name,input_schema={'type':'object'}) for name in
            ['read_file','write_file','list_files','rag_search','read_concept','knowledge_status','sync_knowledge']])
    async def call_tool(self,name,arguments):
        self.calls.append((name,arguments))
        try:
            result=await asyncio.to_thread(getattr(self.tools,name),**arguments)
            return NS(is_error=False,content=[NS(type='text',text=result)])
        except Exception as exc:
            return NS(is_error=True,content=[NS(type='text',text=str(exc))])

def test_scheduler_parallelism_and_bound(project):
    async def scenario():
        config=AgentConfig(allowed_dir=str(project),max_workers=2)
        manager=DelegationManager(project,config)
        active=peak=0
        async def worker(context):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            await asyncio.sleep(0.025)
            active-=1
            return context.task.id
        result=await manager.delegate([{'id':str(i),'instructions':'inspect'} for i in range(5)],TurnBudget(),worker)
        assert peak==2 and active==0
        assert [r['id'] for r in result['workers']]==['0','1','2','3','4']
        assert all(r['state']=='completed' for r in result['workers'])
    asyncio.run(scenario())

@pytest.mark.parametrize('tasks',[
    [{'id':'same','instructions':'a'},{'id':'same','instructions':'b'}],
    [{'id':'a','instructions':'a','mode':'edit','writable_paths':['main.py']},
     {'id':'b','instructions':'b','mode':'edit','writable_paths':['./main.py']}],
    [{'id':'a','instructions':'a','mode':'edit','writable_paths':['new']},
     {'id':'b','instructions':'b','mode':'edit','writable_paths':['new/file.py']}],
    [{'id':'a','instructions':'a','mode':'edit','writable_paths':['../escape.py']}],
    [{'id':'a','instructions':'a','mode':'edit','writable_paths':['.env']}],
    [{'id':'a','instructions':'a','mode':'edit','writable_paths':[]}],
    [{'id':'a','instructions':'a','writable_paths':['main.py']}],
    [{'id':'a','instructions':' '}],
])
def test_invalid_assignments_rejected_before_work(project,tasks):
    async def scenario():
        manager=DelegationManager(project,AgentConfig(allowed_dir=str(project)))
        calls=[]
        async def worker(context):calls.append(context);return 'unexpected'
        with pytest.raises((ValueError,WorkerPolicyError)):
            await manager.delegate(tasks,TurnBudget(),worker)
        assert not calls
    asyncio.run(scenario())

def test_turn_budget_and_disable(project):
    async def scenario():
        config=AgentConfig(allowed_dir=str(project))
        manager=DelegationManager(project,config);budget=TurnBudget(batches=1,tasks=1)
        async def worker(context):return 'done'
        await manager.delegate([{'id':'a','instructions':'inspect'}],budget,worker)
        with pytest.raises(WorkerPolicyError,match='budget'):
            await manager.delegate([{'id':'b','instructions':'inspect'}],budget,worker)
        config.enable_delegation=False
        with pytest.raises(WorkerPolicyError,match='disabled'):
            await manager.delegate([{'id':'c','instructions':'inspect'}],TurnBudget(),worker)
    asyncio.run(scenario())

def test_timeout_failure_isolation_and_partial_write_reporting(project):
    async def scenario():
        manager=DelegationManager(project,AgentConfig(allowed_dir=str(project),worker_timeout_seconds=.04))
        stopped=asyncio.Event()
        async def worker(context):
            if context.task.id=='slow':
                context.record['attempted_paths'].append('uncertain.py')
                try:await asyncio.sleep(5)
                finally:stopped.set()
            if context.task.id=='bad':
                context.record['written_paths'].append('already-written.py')
                raise RuntimeError('injected provider failure')
            return 'good'
        report=await manager.delegate([{'id':x,'instructions':'inspect'} for x in ['slow','bad','good']],TurnBudget(),worker)
        assert [r['state'] for r in report['workers']]==['timeout','failed','completed']
        assert stopped.is_set()
        assert report['workers'][0]['unconfirmed_paths']==['uncertain.py']
        assert report['workers'][1]['written_paths']==['already-written.py']
    asyncio.run(scenario())

def test_parent_cancellation_stops_workers(project):
    async def scenario():
        manager=DelegationManager(project,AgentConfig(allowed_dir=str(project),max_workers=2))
        ready=asyncio.Event();running=0;stopped=[]
        async def worker(context):
            nonlocal running
            running+=1
            if running==2:ready.set()
            try:await asyncio.sleep(5)
            finally:stopped.append(context.task.id)
        task=asyncio.create_task(manager.delegate([{'id':x,'instructions':'inspect'} for x in ['a','b']],TurnBudget(),worker))
        await asyncio.wait_for(ready.wait(),1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert set(stopped)=={'a','b'}
        assert all(r['state']=='cancelled' for r in manager.status()['workers'])
    asyncio.run(scenario())

@pytest.mark.parametrize('mode,name,path',[('read_only','write_file','main.py'),
    ('edit','write_file','main.py'),('read_only','delegate_tasks',None)])
def test_worker_cannot_escape_tool_or_write_scope(project,monkeypatch,mode,name,path):
    async def scenario():
        async def handler(**kwargs):
            assert 'delegate_tasks' not in [t['function']['name'] for t in kwargs['tools']]
            return message(calls=[call(name,{'path':path,'content':'overwrite'} if path else {'tasks':[]})])
        agent=make_agent(project,monkeypatch,handler)
        session=LocalSession(project)
        spec={'id':'guard','instructions':'inspect','mode':mode,'writable_paths':['owned.py'] if mode=='edit' else []}
        report=await agent.delegation.delegate([spec],TurnBudget(),lambda c:agent._run_worker(session,c))
        assert report['workers'][0]['state']=='policy_error'
        assert not session.calls
    asyncio.run(scenario())

def test_external_edit_is_preserved_on_worker_write_conflict(project,monkeypatch):
    async def scenario():
        path=project/'owned.py';path.write_text('before')
        async def handler(**kwargs):
            path.write_text('external user edit')
            return message(calls=[call('write_file',{'path':'owned.py','content':'worker edit','expected_sha256':digest('external user edit')})])
        agent=make_agent(project,monkeypatch,handler);session=LocalSession(project)
        report=await agent.delegation.delegate([{'id':'edit','instructions':'edit','mode':'edit','writable_paths':['owned.py']}],
            TurnBudget(),lambda c:agent._run_worker(session,c))
        assert report['workers'][0]['state']=='conflict'
        assert not report['workers'][0]['written_paths']
        assert path.read_text()=='external user edit'
    asyncio.run(scenario())

def test_worker_context_isolation_round_and_result_limits(project,monkeypatch):
    async def scenario():
        async def handler(**kwargs):
            assert 'MASTER_SECRET_CONTEXT' not in json.dumps(kwargs['messages'])
            assert kwargs['max_tokens']==2048
            return message('é'*2000)
        agent=make_agent(project,monkeypatch,handler,worker_result_bytes=256)
        report=await agent.delegation.delegate([{'id':'a','instructions':'inspect'}],TurnBudget(),
            lambda c:agent._run_worker(LocalSession(project),c))
        assert 'truncated' in report['workers'][0]['summary']
        assert report['workers'][0]['input_tokens']==5
        async def repeat(**kwargs):return message(calls=[call('read_file',{'path':'main.py'})])
        agent.groq_client.chat.completions.create=repeat
        agent.config.max_worker_rounds=1
        report=await agent.delegation.delegate([{'id':'a','instructions':'inspect'}],TurnBudget(),
            lambda c:agent._run_worker(LocalSession(project),c))
        assert report['workers'][0]['state']=='failed' and report['workers'][0]['llm_calls']==1
    asyncio.run(scenario())

def test_coordinator_parallel_workers_share_real_mcp_and_sync(kb,monkeypatch):
    """Real MCP writes, isolated model histories, coordinator merge and watcher refresh."""
    async def scenario():
        from rag import SyncService
        service=SyncService(kb,debounce=.03,reconcile_seconds=.2).start()
        master_calls=0;waiting=set();overlap=asyncio.Event()
        async def handler(**kwargs):
            nonlocal master_calls
            system=kwargs['messages'][0]['content']
            if 'You are the coordinator.' in system:
                master_calls+=1
                if master_calls==1:
                    return message(calls=[call('delegate_tasks',{'tasks':[
                        {'id':x,'instructions':f'TASK_{x}: create {x}.py only','mode':'edit','writable_paths':[x+'.py']} for x in ('alpha','beta')]})])
                result=json.loads(kwargs['messages'][-1]['content'])
                assert all(w['state']=='completed' for w in result['workers'])
                assert all(w['written_paths']==[w['id']+'.py'] for w in result['workers'])
                return message('Both delegated files created and verified.')
            user=kwargs['messages'][1]['content']
            identity='alpha' if 'TASK_alpha' in user else 'beta'
            other='beta' if identity=='alpha' else 'alpha'
            assert 'MASTER_SECRET_CONTEXT' not in json.dumps(kwargs['messages'])
            assert f'TASK_{other}' not in json.dumps(kwargs['messages'])
            assert 'delegate_tasks' not in [t['function']['name'] for t in kwargs['tools']]
            if len(kwargs['messages'])==2:
                waiting.add(identity)
                if len(waiting)==2:overlap.set()
                await asyncio.wait_for(overlap.wait(),3)
                return message(calls=[call('write_file',{'path':identity+'.py','content':f'def {identity}(): return 1\n'})])
            if len(kwargs['messages'])==4:
                return message(calls=[call('read_file',{'path':identity+'.py'})])
            return message(f'{identity}.py created and read back; no execution claimed.')
        agent=make_agent(kb.root,monkeypatch,handler,max_workers=2)
        try:
            async with agent.connect():
                original_session=agent._session
                answer=await agent.run([{'role':'user','content':'MASTER_SECRET_CONTEXT: create two independent files.'}])
                assert agent._session is original_session and 'Both delegated' in answer
                assert len(waiting)==2
            for _ in range(100):
                if kb.status()['indexed']==6:break
                await asyncio.sleep(.05)
            assert kb.status()['indexed']==6
            assert all((kb.root/(x+'.py')).is_file() for x in ('alpha','beta'))
        finally:service.stop();service.join()
    asyncio.run(scenario())

def test_rate_limit_retry_preserves_completed_workers(project,monkeypatch):
    async def scenario():
        class Limited(Exception):
            response=NS(headers={'retry-after':'0'})
        monkeypatch.setattr('agent.RateLimitError',Limited)
        masters=0;workers=[]
        async def handler(**kwargs):
            nonlocal masters
            if 'You are the coordinator.' in kwargs['messages'][0]['content']:
                masters+=1
                if masters==1:
                    return message(calls=[call('delegate_tasks',{'tasks':[
                        {'id':x,'instructions':x} for x in ('a','b')]})])
                if masters==2:raise Limited('injected quota')
                report=json.loads(kwargs['messages'][-1]['content'])
                assert all(r['state']=='completed' for r in report['workers'])
                return message('Merged existing worker results')
            workers.append(kwargs['messages'][1]['content'])
            return message('Findings')
        agent=make_agent(project,monkeypatch,handler)
        agent._session=LocalSession(project)
        assert 'Merged' in await agent.run([{'role':'user','content':'inspect'}])
        assert masters==3 and sorted(workers)==['a','b']
        assert agent.agent_status()['rate_limit_retries']==1
    asyncio.run(scenario())

def test_rate_limit_wait_is_bounded(project,monkeypatch):
    async def scenario():
        class Limited(Exception):
            response=NS(headers={'retry-after':'90'})
        monkeypatch.setattr('agent.RateLimitError',Limited)
        async def handler(**kwargs):raise Limited('wait')
        agent=make_agent(project,monkeypatch,handler)
        with pytest.raises(RuntimeError,match='wait budget'):
            await agent._completion(model='test',messages=[])
    asyncio.run(scenario())
