"""Interactive client; the persistent MCP process owns the model and watcher."""
import os
import argparse
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from agent import CodebaseGuardianAgent, AgentConfig
from workspace import validate_agent_workspace

load_dotenv(Path(__file__).resolve().parent/'.env')
console = Console()

async def main():
    parser=argparse.ArgumentParser(description='AgentoCode CLI')
    parser.add_argument('--dir',required=True)
    parser.add_argument('--workers',type=int,default=3,choices=range(1,7),help='Maximum concurrent delegated workers (default: 3)')
    parser.add_argument('--single-agent',action='store_true',help='Disable worker delegation')
    parser.add_argument('--worker-model',help='Optional Groq worker model; defaults to the coordinator model')
    args=parser.parse_args()
    target=str(validate_agent_workspace(args.dir))
    if not os.getenv('GROQ_API_KEY'):
        raise SystemExit('GROQ_API_KEY is missing from the environment or .env')
    agent=CodebaseGuardianAgent(AgentConfig(allowed_dir=target,max_workers=args.workers,
        enable_delegation=not args.single_agent,worker_model=args.worker_model))
    session=PromptSession(history=FileHistory(str(Path(target)/'.cg_history')))
    history=[]
    console.print(f'AgentoCode workspace: {target}',markup=False)
    console.print('Starting the knowledge worker. Use /status for loading/indexing progress.')
    async with agent.connect(watch=True):
        console.print('Commands: /status, /agents, /sync, /clear, /exit')
        while True:
            try:
                value=(await session.prompt_async('\n> ')).strip()
                if not value:
                    continue
                if value.lower() in ('/exit','/quit'):
                    break
                if value.lower() == '/clear':
                    history.clear()
                    console.print('Chat history cleared.')
                    continue
                if value.lower() == '/sync':
                    console.print(await agent.workspace_tool('sync_knowledge'),markup=False)
                    continue
                if value.lower() == '/status':
                    console.print_json(await agent.workspace_tool('knowledge_status'))
                    continue
                if value.lower() == '/agents':
                    console.print_json(json.dumps(agent.agent_status()))
                    continue
                history.append({'role':'user','content':value})
                try:
                    with console.status('Agent is investigating...'):
                        answer=await agent.run(history)
                except Exception as exc:
                    answer=f'Agent error: {exc}'
                console.print(Markdown(answer))
                history.append({'role':'assistant','content':answer})
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
    console.print('AgentoCode stopped.')

if __name__ == '__main__':
    asyncio.run(main())
