import os
import argparse
import subprocess
from mcp.server.fastmcp import FastMCP
from rag import RAGManager

def parse_args():
    parser = argparse.ArgumentParser(description="MCP Server for Codebase Guardian")
    parser.add_argument("--allowed-dir", type=str, required=True, help="Allowed directory for sandboxing")
    return parser.parse_args()

args = parse_args()
ALLOWED_DIR = os.path.abspath(args.allowed_dir)

# Initialize FastMCP Server
mcp = FastMCP("CodebaseGuardian")

# Initialize RAGManager (this just sets paths, the Chroma DB is lazily loaded on search)
rag_manager = RAGManager(ALLOWED_DIR)

def ensure_safe_path(file_path: str) -> str:
    """Ensures the path is within the ALLOWED_DIR."""
    abs_path = os.path.abspath(os.path.join(ALLOWED_DIR, file_path))
    if not abs_path.startswith(ALLOWED_DIR):
        raise ValueError(f"Access denied: path {abs_path} is outside allowed directory {ALLOWED_DIR}")
    return abs_path

@mcp.tool()
def read_file(path: str) -> str:
    """Reads the content of a file in the workspace."""
    try:
        safe_path = ensure_safe_path(path)
        with open(safe_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Writes content to a file in the workspace."""
    try:
        safe_path = ensure_safe_path(path)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {safe_path}"
    except Exception as e:
        return f"Error writing file: {e}"

@mcp.tool()
def run_command(command: str) -> str:
    """Runs a terminal command in the workspace directory."""
    try:
        # Run command with cwd set to ALLOWED_DIR
        result = subprocess.run(
            command,
            cwd=ALLOWED_DIR,
            shell=True,
            capture_output=True,
            text=True
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output if output else "Command executed successfully with no output."
    except Exception as e:
        return f"Error running command: {e}"

@mcp.tool()
def rag_search(query: str) -> str:
    """Searches the codebase for semantic context using RAG."""
    try:
        return rag_manager.search(query)
    except Exception as e:
        return f"Error searching codebase: {e}"

if __name__ == "__main__":
    mcp.run()
