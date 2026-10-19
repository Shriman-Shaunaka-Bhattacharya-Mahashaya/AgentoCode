import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import pytest
from agent import CodebaseGuardianAgent, AgentConfig
from server import WorkspaceTools, create_server
from workspace import APP_ROOT, safe_path, inventory
from rag import SyncService
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.session import ClientSession

def test_agent_refuses_own_root_and_ancestors():
    for root in (APP_ROOT,APP_ROOT.parent):
        with pytest.raises(ValueError,match='agent application'):
            WorkspaceTools(root)

def test_file_tool_escape_and_reserved_paths(project):
    tools = WorkspaceTools(project)
    for value in ('../agent.py', str(APP_ROOT/'agent.py'), '.env', '.guardian/x', 'knowledge_base/x'):
        with pytest.raises(ValueError):
            tools.write_file(value,'denied')
    with pytest.raises(ValueError):
        tools.read_file('../project-other/file.py')

def test_junction_escape_rejected(project):
    # Windows junctions need no Developer Mode; target remains inside the fixture parent.
    outside = project.parent/'outside'
    outside.mkdir()
    (outside/'secret.py').write_text('secret')
    junction = project/'linked'
    if sys.platform == 'win32':
        subprocess.run(['cmd','/c','mklink','/J',str(junction),str(outside)],check=True,capture_output=True)
    else:
        junction.symlink_to(outside,target_is_directory=True)
    try:
        with pytest.raises(ValueError):
            WorkspaceTools(project).read_file('linked/secret.py')
        assert 'linked/secret.py' not in inventory(project)[0]
    finally:
        if sys.platform == 'win32':
            junction.rmdir()
        else:
            junction.unlink()

def test_real_mcp_sample_project_read_and_escape():
    async def scenario():
        params = StdioServerParameters(command=sys.executable,args=[str(APP_ROOT/'server.py'),'--allowed-dir',str(APP_ROOT/'sample_project')])
        async with stdio_client(params) as (read,write):
            async with ClientSession(read,write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                assert 'run_command' not in names
                assert {'read_file','write_file','rag_search','list_files','read_concept'} <= names
                result = await session.call_tool('read_file',{'path':'calculator.py'})
                assert 'Cannot divide by zero' in ''.join(c.text for c in result.content if c.type=='text')
                escaped = await session.call_tool('write_file',{'path':'../agent.py','content':'DENIED'})
                assert escaped.is_error
    before = (APP_ROOT/'agent.py').read_bytes()
    asyncio.run(scenario())
    assert (APP_ROOT/'agent.py').read_bytes() == before

def test_agent_loop_on_real_sample_project(monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY','deterministic-test-key-not-a-credential')
    guardian = CodebaseGuardianAgent(AgentConfig(allowed_dir=str(APP_ROOT/'sample_project')))
    calls = []
    class Completions:
        async def create(self,**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                return NS(choices=[NS(message=NS(content=None,tool_calls=[NS(id='read1',function=NS(name='read_file',arguments='{"path":"calculator.py"}'))]))])
            tool_result = kwargs['messages'][-1]['content']
            assert 'Raises ValueError' in tool_result
            return NS(choices=[NS(message=NS(content='Division by zero raises ValueError.',tool_calls=None))])
    guardian.groq_client = NS(chat=NS(completions=Completions()))
    sources = {p:p.read_bytes() for p in APP_ROOT.glob('*.py')}
    answer = asyncio.run(guardian.run([{'role':'user','content':'Explain the calculator division error.'}]))
    assert answer == 'Division by zero raises ValueError.'
    assert len(calls)==2
    assert all(p.read_bytes()==raw for p,raw in sources.items())

def test_real_watcher_user_and_agent_updates(kb):
    service = SyncService(kb,debounce=0.05,reconcile_seconds=0.5).start()
    def wait_for(predicate):
        end = time.monotonic()+15
        while time.monotonic()<end:
            try:
                if predicate(): return
            except (KeyError,OSError,ValueError): pass
            time.sleep(0.05)
        raise AssertionError('Watcher did not converge')
    try:
        wait_for(lambda:len(kb.index.records)==4)
        counts = len(kb.index.calls)
        (kb.root/'calculator.py').write_text('def changed_user(): return 1\n')
        wait_for(lambda: len(kb.index.calls)>counts and kb.status()['indexed']==4)
        counts = len(kb.index.calls)
        WorkspaceTools(kb.root,kb).write_file('new_agent.py','def changed_agent(): return 2\n')
        wait_for(lambda:kb.record_id('file','new_agent.py') in kb.index.records and kb.status()['indexed']==5)
    finally:
        service.stop()
        service.join()
    assert not service.thread.is_alive()

def test_sample_calculator_executes_without_agent_edits():
    before = {p:p.read_bytes() for p in (APP_ROOT/'sample_project').glob('*') if p.is_file()}
    result = subprocess.run([sys.executable,str(APP_ROOT/'sample_project'/'main.py')],cwd=APP_ROOT/'sample_project',capture_output=True,text=True,timeout=20)
    assert result.returncode==0
    assert '5! = 120' in result.stdout and 'Cannot divide by zero' in result.stdout
    assert all(p.read_bytes()==raw for p,raw in before.items())
