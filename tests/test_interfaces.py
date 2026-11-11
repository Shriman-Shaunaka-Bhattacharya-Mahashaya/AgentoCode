"""Public CLI/tool contracts, including failures and cleanup, without live providers."""
import asyncio
from contextlib import asynccontextmanager
import json
import sys
from types import SimpleNamespace as NS
import pytest
from server import WorkspaceTools
from workspace import digest

@pytest.mark.parametrize('ending',['/exit','eof'])
def test_cli_commands_flags_history_and_provider_error(project,monkeypatch,capsys,ending):
    import cli
    instances=[]
    class Agent:
        def __init__(self,config):self.config=config;self.closed=False;self.calls=[];instances.append(self)
        @asynccontextmanager
        async def connect(self,watch=False):
            assert watch
            try:yield self
            finally:self.closed=True
        async def workspace_tool(self,name):return json.dumps({'indexed':4}) if name=='knowledge_status' else 'sync queued'
        def agent_status(self):return {'workers':[]}
        async def run(self,history):
            self.calls.append([m.copy() for m in history])
            if len(self.calls)==1:raise RuntimeError('injected provider failure')
            return 'source verified'
    values=iter(['',KeyboardInterrupt(),'first question','/status','/agents','/sync','/clear','second question',ending])
    class Prompt:
        def __init__(self,**kwargs):pass
        async def prompt_async(self,*args):
            value=next(values)
            if isinstance(value,BaseException):raise value
            if value=='eof':raise EOFError
            return value
    monkeypatch.setenv('GROQ_API_KEY','test-only-placeholder')
    monkeypatch.setattr(cli,'CodebaseGuardianAgent',Agent)
    monkeypatch.setattr(cli,'PromptSession',Prompt)
    monkeypatch.setattr(sys,'argv',['cli.py','--dir',str(project),'--workers','2','--single-agent','--worker-model','fixture-model'])
    asyncio.run(cli.main())
    agent=instances[0]
    assert agent.closed and agent.config.max_workers==2 and not agent.config.enable_delegation
    assert agent.config.worker_model=='fixture-model'
    assert len(agent.calls[1])==1 and agent.calls[1][0]['content']=='second question'
    out=capsys.readouterr().out
    assert 'injected provider failure' in out and 'source verified' in out and 'AgentoCode stopped' in out

def test_cli_missing_key_and_invalid_worker_count(project,monkeypatch):
    import cli
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    monkeypatch.setattr(sys,'argv',['cli.py','--dir',str(project)])
    with pytest.raises(SystemExit,match='GROQ_API_KEY'):asyncio.run(cli.main())
    monkeypatch.setattr(sys,'argv',['cli.py','--dir',str(project),'--workers','0'])
    with pytest.raises(SystemExit) as failure:asyncio.run(cli.main())
    assert failure.value.code==2

def test_file_tool_limits_hash_guards_and_sync_notifications(kb):
    tools=WorkspaceTools(kb.root,kb);notices=[]
    tools.service=NS(request_sync=lambda:notices.append('sync'))
    tools.write_file('guard.py','value=1\n',expected_sha256='missing')
    tools.write_file('guard.py','value=2\n',expected_sha256=digest('value=1\n'))
    with pytest.raises(ValueError,match='Write conflict'):
        tools.write_file('guard.py','overwrite',expected_sha256='missing')
    assert (kb.root/'guard.py').read_text()=='value=2\n' and len(notices)==2
    assert 'queued' in tools.sync_knowledge()
    tools.service=None;assert 'No background worker' in tools.sync_knowledge()
    with pytest.raises(ValueError,match='write limit'):tools.write_file('large.txt','x'*512001)
    (kb.root/'binary.txt').write_bytes(b'a\0b')
    with pytest.raises(ValueError,match='Binary'):tools.read_file('binary.txt')
    (kb.root/'large.txt').write_text('x'*512001)
    with pytest.raises(ValueError,match='read limit'):tools.read_file('large.txt')
    (kb.root/'lines.txt').write_text('x\n'*300)
    assert len(tools.read_file('lines.txt',start_line=-1,max_lines=999).splitlines())==200
    assert 'guard.py' in json.loads(tools.list_files())['files']

def test_concept_navigation_freshness_validation_and_partial_output(kb):
    tools=WorkspaceTools(kb.root,kb)
    assert 'absent or stale' in tools.read_concept('calculator.py')
    kb.sync()
    assert '# Explanation' in tools.read_concept('calculator.py')
    assert '# Explanation' in tools.read_concept('','directory')
    with pytest.raises(ValueError):tools.read_concept('calculator.py','unknown')
    record=kb.load()['records'][kb.record_id('file','calculator.py')]
    path=kb.bundle/record['concept']
    path.write_text(path.read_text(encoding='utf-8')+'\n'+'extended explanation\n'*1200,encoding='utf-8')
    assert 'unvalidated' in tools.read_concept('calculator.py')
    kb.sync()
    result=tools.read_concept('calculator.py')
    assert 'Partial coverage' in result and len(result.encode('utf-8'))<=16000
    (kb.root/'calculator.py').write_text('def current(): return 1\n')
    assert 'absent or stale' in tools.read_concept('calculator.py')

def test_model_cache_identity_and_checksum_fail_closed(tmp_path):
    from embeddings import Profile,SentenceTransformersEmbedder
    profile=Profile();folder=tmp_path/profile.name/profile.revision
    folder.mkdir(parents=True);(folder/'weights').write_bytes(b'original')
    manifest={'model':profile.model,'revision':profile.revision,'files':{'weights':'0'*64}}
    (folder/'asset-manifest.json').write_text(json.dumps(manifest))
    factory_calls=[]
    embedder=SentenceTransformersEmbedder(model_factory=lambda *a,**kw:factory_calls.append((a,kw)))
    embedder.asset_root=tmp_path
    with pytest.raises(ValueError,match='checksum'):embedder.embed_documents(['text'])
    assert not factory_calls and embedder.status()['state']=='failed'
    manifest['revision']='wrong';(folder/'asset-manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity'):embedder.embed_queries(['text'])
    assert not factory_calls

def test_offline_model_fallback_and_bad_response_count(monkeypatch):
    from test_embeddings import StubEncoder
    original=StubEncoder();calls=[];factory=original.model_factory
    def capture(*args,**kwargs):calls.append(kwargs);return factory(*args,**kwargs)
    original.model_factory=capture
    monkeypatch.setenv('GUARDIAN_LOCAL_FILES_ONLY','1')
    original.embed_documents(['text'])
    assert calls[0]['local_files_only'] is True and calls[0]['trust_remote_code'] is False
    original.model.encode=lambda *a,**kw:[]
    with pytest.raises(ValueError,match='count mismatch'):original.embed_queries(['text'])

def test_wrong_model_context_width_and_pooling_rejected(monkeypatch):
    from test_embeddings import StubEncoder
    for problem in ('context','width','pooling'):
        embedder=StubEncoder();factory=embedder.model_factory
        def broken(*args,_factory=factory,_problem=problem,**kwargs):
            model=_factory(*args,**kwargs)
            if _problem=='context':model.max_seq_length=1
            elif _problem=='width':model.get_sentence_embedding_dimension=lambda:7
            else:
                original=type(model).__getitem__
                monkeypatch.setattr(type(model),'__getitem__',lambda self,i: NS(pooling_mode='max') if i==1 else original(self,i))
            return model
        embedder.model_factory=broken
        with pytest.raises(ValueError):embedder.embed_documents(['text'])
        assert embedder.load_count==0
