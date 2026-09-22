import asyncio
from dotenv import load_dotenv
load_dotenv()
from agent import CodebaseGuardianAgent, AgentConfig

async def test():
    config = AgentConfig(allowed_dir=".\\sample_project")
    agent = CodebaseGuardianAgent(config)
    
    chat_history = [{"role": "user", "content": "What files are in the directory?"}]
    response = await agent.run(chat_history)
    print("Agent Response:", response)

if __name__ == "__main__":
    # Ensure asyncio uses the correct event loop policy on Windows for subprocesses (MCP)
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(test())
