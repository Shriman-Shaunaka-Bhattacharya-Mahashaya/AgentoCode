"""Boundary/failure cases discovered during the complete validation pass."""
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import math
import pytest
from agent import AgentConfig
from delegation import DelegationManager, TurnBudget, WorkerPolicyError
from embeddings import normalized
from generation import GroqGenerator, logical_units
from workspace import inventory, safe_path

def test_shared_turn_budget_is_rechecked_after_batch_lock(project):
    async def scenario():
        manager=DelegationManager(project,AgentConfig(allowed_dir=str(project)))
        budget=TurnBudget(batches=1,tasks=1);executed=[]
        async def worker(context):executed.append(context.task.id);return 'done'
        await manager.batch_lock.acquire()
        tasks=[asyncio.create_task(manager.delegate([{'id':name,'instructions':'inspect'}],budget,worker)) for name in ('a','b')]
        await asyncio.sleep(0)
        manager.batch_lock.release()
        results=await asyncio.gather(*tasks,return_exceptions=True)
        assert len(executed)==1
        assert sum(isinstance(r,WorkerPolicyError) for r in results)==1
        assert budget.tasks==0 and budget.batches==0
    asyncio.run(scenario())

def test_logical_units_do_not_duplicate_same_line_statements():
    text='a = 1; b = 2\n# trailing note\n'
    assert ''.join(logical_units('module.py',text))==text

@pytest.mark.parametrize('value',[1e308,1e-300])
def test_normalization_handles_extreme_finite_nonzero_vectors(value):
    result=normalized([value,value],2)
    assert all(math.isfinite(x) for x in result)
    assert math.isclose(math.hypot(*result),1,rel_tol=1e-12)

def test_non_git_fallback_cannot_reinclude_inside_ignored_parent(project,monkeypatch):
    import workspace
    (project/'.gitignore').write_text('private/\n')
    (project/'private').mkdir()
    (project/'private/.gitignore').write_text('!secret.txt\n')
    (project/'private/secret.txt').write_text('fixture only')
    monkeypatch.setattr(workspace.subprocess,'run',lambda *a,**kw: (_ for _ in ()).throw(OSError('git unavailable')))
    files,dirs,_=inventory(project)
    assert 'private/secret.txt' not in files and 'private' not in dirs

def test_generated_directory_exclusions_are_case_insensitive(project):
    (project/'VENV').mkdir();(project/'VENV/generated.py').write_text('pass')
    assert 'VENV/generated.py' not in inventory(project)[0]

@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows filename aliases/streams')
@pytest.mark.parametrize('path',['main.py:alternate','NUL','CON.txt','main.py.','main.py '])
def test_windows_ambiguous_or_device_paths_rejected_before_io(project,path):
    with pytest.raises(ValueError):safe_path(project,path)

def test_corrupt_generation_cache_regenerates(tmp_path):
    calls=[]
    class Completions:
        def create(self,**kwargs):
            calls.append(kwargs)
            return NS(choices=[NS(message=NS(content='{"summary":"fresh"}'),finish_reason='stop')])
    gen=GroqGenerator(client=NS(chat=NS(completions=Completions())),retries=0)
    gen.cache_dir=tmp_path/'cache'
    assert gen.explain('file','a.py','x=1')=='fresh'
    path=next(gen.cache_dir.glob('*.json'));path.write_text('{broken',encoding='utf-8')
    assert gen.explain('file','a.py','x=1')=='fresh'
    assert len(calls)==2

def test_truncated_generation_is_not_published_as_complete():
    class Completions:
        def create(self,**kwargs):
            return NS(choices=[NS(message=NS(content='{"summary":"incomplete"}'),finish_reason='length')])
    gen=GroqGenerator(client=NS(chat=NS(completions=Completions())),retries=0)
    with pytest.raises(ValueError,match='truncat'):gen.explain('file','a.py','x=1')

def test_stale_semantic_records_are_not_counted_as_current_candidates(kb):
    kb.sync()
    rid=kb.record_id('file','calculator.py')
    (kb.root/'calculator.py').write_text('def latest(): return 9\n')
    kb.index.query=lambda *a,**kw:[(rid,.01)]
    result=kb.search('unrelated_token_x9',structured=True)
    assert result['semantic_candidate_count']==0
    assert result['retrieval_mode']=='source_fallback'

def test_status_does_not_label_changed_source_current_before_sync(kb):
    kb.sync()
    (kb.root/'calculator.py').write_text('def latest(): return 9\n')
    status=kb.status()
    assert status['indexed']<status['concepts']
    assert status.get('stale',0)>=1

def test_downloader_recovers_same_size_corrupt_cached_weights(tmp_path,monkeypatch):
    import download_model
    from embeddings import Profile
    monkeypatch.setattr(download_model,'__file__',str(tmp_path/'download_model.py'))
    profile=Profile();root=tmp_path/'.guardian/models'/profile.name/profile.revision
    root.mkdir(parents=True)
    weights=bytes(range(64));sha=hashlib.sha256(weights).hexdigest()
    (root/'model.safetensors').write_bytes(b'x'*len(weights))
    def curl(command,**kwargs):
        if '--head' in command:
            return NS(returncode=0,stdout=f'Content-Length: {len(weights)}\nx-linked-etag: "{sha}"\n',stderr='')
        destination=Path(command[command.index('-o')+1])
        if '--range' in command:
            start,end=map(int,command[command.index('--range')+1].split('-'))
            destination.write_bytes(weights[start:end+1])
        else:destination.write_text('{}')
        return NS(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(download_model.subprocess,'run',curl)
    download_model.download()
    assert (root/'model.safetensors').read_bytes()==weights
    assert (root/'asset-manifest.json').is_file()

@pytest.mark.parametrize('header',['inf','nan',None,'invalid-date'])
def test_malformed_rate_limit_header_does_not_poison_later_requests(project,monkeypatch,header):
    from test_delegation import make_agent,message
    async def scenario():
        class Limited(Exception):response=NS(headers={'retry-after':header})
        monkeypatch.setattr('agent.RateLimitError',Limited)
        calls=[]
        async def handler(**kwargs):
            calls.append(kwargs)
            if len(calls)==1:raise Limited('Please try again in 0.01s.')
            return message('ok')
        agent=make_agent(project,monkeypatch,handler)
        assert (await agent._completion(model='fixture',messages=[])).choices[0].message.content=='ok'
        assert math.isfinite(agent._provider_resume_at) and len(calls)==2
    asyncio.run(scenario())

def test_http_date_rate_limit_header_is_supported():
    from datetime import datetime,timezone,timedelta
    from email.utils import format_datetime
    from agent import CodebaseGuardianAgent
    header=format_datetime(datetime.now(timezone.utc)+timedelta(seconds=5))
    delay=CodebaseGuardianAgent._retry_delay(NS(response=NS(headers={'retry-after':header})),0)
    assert 3 < delay <= 5.25

@pytest.mark.parametrize('persistent',[False,True])
def test_manifest_reads_retry_sharing_errors_without_hiding_permanent_failure(kb,monkeypatch,persistent):
    import knowledge
    kb.sync();path=kb.bundle/'.manifest.json';original_bytes=path.read_bytes()
    original=Path.read_text;attempts=[]
    def read(self,*args,**kwargs):
        if self==path:
            attempts.append(self)
            if persistent or len(attempts)<3:raise PermissionError('injected sharing violation')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr(knowledge.time,'sleep',lambda _:None)
    if persistent:
        with pytest.raises(PermissionError):kb.load()
        assert len(attempts)==6
    else:
        assert len(kb.load()['records'])==4 and len(attempts)==3
    assert path.read_bytes()==original_bytes
