"""Coordinator public failure contracts and uncertain-write reporting."""
import asyncio
import json
from types import SimpleNamespace as NS
import pytest
from delegation import TurnBudget
from test_delegation import make_agent,LocalSession,call,message

def test_disabled_delegation_removes_coordinator_tools(project,monkeypatch):
    async def scenario():
        async def handler(**kwargs):
            names={tool['function']['name'] for tool in kwargs['tools']}
            assert 'delegate_tasks' not in names and 'worker_status' not in names
            return message('single-agent response')
        agent=make_agent(project,monkeypatch,handler,enable_delegation=False)
        agent._session=LocalSession(project)
        assert await agent.run([{'role':'user','content':'inspect'}])=='single-agent response'
        result=await agent._execute_tool(agent._session,call('delegate_tasks',{'tasks':[{'id':'a','instructions':'inspect'}]}),TurnBudget())
        assert 'disabled' in result and not agent.delegation.last_batch
    asyncio.run(scenario())

def test_coordinator_input_budget_rejects_before_provider_call(project,monkeypatch):
    async def scenario():
        calls=[]
        async def handler(**kwargs):calls.append(kwargs);return message('unexpected')
        agent=make_agent(project,monkeypatch,handler,max_input_bytes=100)
        agent._session=LocalSession(project)
        with pytest.raises(RuntimeError,match='input budget'):
            await agent.run([{'role':'user','content':'x'*101}])
        assert not calls
    asyncio.run(scenario())

def test_coordinator_model_rounds_are_bounded(project,monkeypatch):
    async def scenario():
        calls=[]
        async def handler(**kwargs):
            calls.append(kwargs)
            return message(calls=[call('read_file',{'path':'main.py'},f'read{len(calls)}')])
        agent=make_agent(project,monkeypatch,handler)
        agent._session=LocalSession(project)
        with pytest.raises(RuntimeError,match='20 model rounds'):
            await agent.run([{'role':'user','content':'inspect'}])
        assert len(calls)==20
    asyncio.run(scenario())

def test_tool_argument_status_and_transport_failures_are_reported(project,monkeypatch):
    async def scenario():
        async def handler(**kwargs):return message('unused')
        agent=make_agent(project,monkeypatch,handler);session=LocalSession(project)
        bad=NS(function=NS(name='read_file',arguments='{broken'))
        assert 'Failed to parse JSON' in await agent._execute_tool(session,bad)
        result=json.loads(await agent._execute_tool(session,call('worker_status',{})))
        assert result['workers']==[]
        assert 'takes no arguments' in await agent._execute_tool(session,call('worker_status',{'unexpected':True}))
        assert 'requires only a tasks array' in await agent._execute_tool(session,call('delegate_tasks',{'wrong':[]}),TurnBudget())
        async def broken(*a,**kw):raise RuntimeError('injected transport failure')
        session.call_tool=broken
        assert 'injected transport failure' in await agent._execute_tool(session,call('read_file',{'path':'main.py'}))
    asyncio.run(scenario())

def test_connection_preconditions_do_not_spawn_another_server(project,monkeypatch):
    async def scenario():
        async def handler(**kwargs):return message('unused')
        agent=make_agent(project,monkeypatch,handler)
        with pytest.raises(RuntimeError,match='Connect'):await agent.workspace_tool('list_files')
        agent._session=LocalSession(project)
        with pytest.raises(RuntimeError,match='already connected'):
            async with agent.connect():pass
    asyncio.run(scenario())

def test_unacknowledged_committed_write_is_reported_as_uncertain(project,monkeypatch):
    async def scenario():
        async def handler(**kwargs):return message(calls=[call('write_file',{'path':'uncertain.py','content':'value = 7\n'})])
        agent=make_agent(project,monkeypatch,handler,worker_timeout_seconds=.1)
        class DelayedAcknowledgement(LocalSession):
            async def call_tool(self,name,arguments):
                result=await super().call_tool(name,arguments)
                assert not result.is_error
                await asyncio.sleep(5)
                return result
        report=await agent.delegation.delegate([{'id':'edit','instructions':'write assigned file',
            'mode':'edit','writable_paths':['uncertain.py']}],TurnBudget(),
            lambda context:agent._run_worker(DelayedAcknowledgement(project),context))
        worker=report['workers'][0]
        assert worker['state']=='timeout' and worker['written_paths']==[]
        assert worker['unconfirmed_paths']==['uncertain.py']
        assert (project/'uncertain.py').read_text()=='value = 7\n'
    asyncio.run(scenario())

def test_provider_attempts_are_bounded(project,monkeypatch):
    async def scenario():
        class Limited(Exception):response=NS(headers={})
        monkeypatch.setattr('agent.RateLimitError',Limited)
        attempts=[]
        async def handler(**kwargs):attempts.append(kwargs);raise Limited('retry')
        agent=make_agent(project,monkeypatch,handler)
        monkeypatch.setattr(agent,'_retry_delay',lambda *a:0)
        with pytest.raises(Limited):await agent._completion(model='fixture',messages=[])
        assert len(attempts)==5 and agent.rate_limit_retries==4
    asyncio.run(scenario())
