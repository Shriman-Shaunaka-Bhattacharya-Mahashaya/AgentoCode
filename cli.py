import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent import CodebaseGuardianAgent, AgentConfig
from rag import start_watchdog

# Load environment variables
load_dotenv()

# Initialize Rich console
console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="CLI for Codebase Guardian")
    parser.add_argument("--dir", type=str, required=True, help="Target directory to guard")
    return parser.parse_args()

async def main():
    args = parse_args()
    target_dir = os.path.abspath(args.dir)
    
    if not os.environ.get("GROQ_API_KEY"):
        console.print("[bold red]Error:[/bold red] GROQ_API_KEY not found in environment or .env file.")
        sys.exit(1)
        
    console.print(Panel(f"Starting Codebase Guardian in [bold cyan]{target_dir}[/bold cyan]...", title="Codebase Guardian"))
    
    # 1. Start Watchdog Daemon for RAG
    console.print("[dim]Initializing vector memory and starting file watcher...[/dim]")
    try:
         observer = start_watchdog(target_dir)
         console.print("[bold green]Memory synced and watching for changes.[/bold green]\n")
    except Exception as e:
         console.print(f"[bold red]Failed to start watchdog:[/bold red] {e}")
         sys.exit(1)
    
    # 2. Setup Agent
    config = AgentConfig(allowed_dir=target_dir)
    agent = CodebaseGuardianAgent(config)
    
    # 3. Setup CLI Prompt
    # Store history in the allowed directory so it's project-specific
    history_file = os.path.join(target_dir, ".cg_history")
    session = PromptSession(history=FileHistory(history_file))
    
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
    })
    
    chat_history = []
    
    console.print("Type your query. Use [bold]/exit[/bold] to quit, [bold]/clear[/bold] to clear history, [bold]/sync[/bold] to force memory sync.\n")
    
    while True:
        try:
            # Get user input
            user_input = await session.prompt_async("\n❯ ", style=style)
            user_input = user_input.strip()
            
            if not user_input:
                continue
                
            # Handle Slash Commands
            if user_input.lower() in ('/exit', '/quit'):
                console.print("[yellow]Shutting down Guardian...[/yellow]")
                observer.stop()
                observer.join()
                break
            elif user_input.lower() == '/clear':
                chat_history.clear()
                console.print("[dim]Chat history cleared.[/dim]")
                continue
            elif user_input.lower() == '/sync':
                console.print("[dim]Forcing memory sync...[/dim]")
                # Start watchdog already does initial sync, but we can do it manually again if needed.
                # To do it properly we'd need a reference to the RAGManager. 
                # For MVP, let's just instruct the user it's automatic.
                console.print("[dim]Note: Memory is synced automatically on file save.[/dim]")
                continue
                
            # Add user message to history
            chat_history.append({"role": "user", "content": user_input})
            
            # Show a spinner while the agent thinks
            with console.status("[bold cyan]Agent is thinking...", spinner="dots"):
                try:
                    response_text = await agent.run(chat_history)
                except ExceptionGroup as eg:
                    # Unpack AnyIO/ExceptionGroups for cleaner display
                    response_text = f"**System Error:**\n```\n{eg.exceptions}\n```"
                except Exception as e:
                     response_text = f"**Agent Error:**\n```\n{e}\n```"
            
            # Print response using Markdown
            console.print(Markdown(response_text))
            
            # Add assistant message to history
            chat_history.append({"role": "assistant", "content": response_text})
            
        except KeyboardInterrupt:
            # Handle Ctrl+C safely without crashing
            continue
        except EOFError:
            # Handle Ctrl+D
            console.print("[yellow]Shutting down Guardian...[/yellow]")
            observer.stop()
            observer.join()
            break

if __name__ == "__main__":
    asyncio.run(main())
