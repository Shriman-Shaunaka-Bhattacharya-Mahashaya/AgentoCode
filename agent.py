import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from groq import AsyncGroq
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import BaseModel
import traceback

class AgentConfig(BaseModel):
    allowed_dir: str
    model: str = "llama-3.1-70b-versatile" # Default model

class CodebaseGuardianAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.system_prompt = (
            "You are an Autonomous Codebase Guardian, an expert AI assistant that helps users understand, "
            "navigate, and modify their codebase. You have access to local file system tools and a semantic "
            "search tool (rag_search) that allows you to find relevant code snippets based on meaning.\n"
            "When asked a question, use your tools to investigate before answering. If a tool fails, "
            "read the error message and try to fix your tool call."
        )

    def _convert_mcp_tool_to_openai_schema(self, mcp_tool) -> Dict[str, Any]:
        """Converts an MCP tool definition to OpenAI's function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.inputSchema
            }
        }

    async def _execute_tool(self, session: ClientSession, tool_call) -> str:
        """Executes a tool call via the MCP session and handles exceptions safely."""
        name = tool_call.function.name
        args_str = tool_call.function.arguments
        
        print(f"\n[Tool Call]: {name}({args_str})", flush=True)
        try:
            # Parse arguments
            args = json.loads(args_str)
            # Call MCP tool
            result = await session.call_tool(name, args)
            
            # Extract content from result
            output = ""
            for content in result.content:
                if content.type == "text":
                    output += content.text
                    
            print(f"[Tool Result]: (Length: {len(output)})\n", flush=True)
            return output
            
        except json.JSONDecodeError as e:
            err = f"Failed to parse JSON arguments for tool {name}: {e}"
            print(f"[Tool Error]: {err}", flush=True)
            return err
        except Exception as e:
            tb = traceback.format_exc()
            err = f"Tool {name} raised an exception:\n{tb}"
            print(f"[Tool Error]:\n{err}", flush=True)
            return err

    async def run(self, chat_history: List[Dict[str, Any]]) -> str:
        """Runs the ReAct loop for a single interaction round."""
        
        # Setup MCP Server connection
        server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.py")
        server_params = StdioServerParameters(
            command="python",
            args=[server_script, "--allowed-dir", self.config.allowed_dir]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Fetch available tools
                mcp_tools_result = await session.list_tools()
                tools = [self._convert_mcp_tool_to_openai_schema(t) for t in mcp_tools_result.tools]
                
                # Prepend system prompt to a copy of history
                messages = [{"role": "system", "content": self.system_prompt}] + chat_history
                
                while True:
                    # Request completion from Groq
                    response = await self.groq_client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=4096
                    )
                    
                    message = response.choices[0].message
                    
                    # Store assistant message
                    # Groq client returns Pydantic objects, we need to convert them appropriately for history
                    assistant_msg = {"role": "assistant"}
                    if message.content:
                        assistant_msg["content"] = message.content
                    if message.tool_calls:
                         # Format tool calls for API
                        tool_calls = []
                        for tc in message.tool_calls:
                            tool_calls.append({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            })
                        assistant_msg["tool_calls"] = tool_calls
                        
                    messages.append(assistant_msg)
                    
                    # If there are no tool calls, we are done
                    if not message.tool_calls:
                        return message.content or ""
                    
                    # Execute tool calls
                    for tool_call in message.tool_calls:
                        tool_result_str = await self._execute_tool(session, tool_call)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result_str
                        })
