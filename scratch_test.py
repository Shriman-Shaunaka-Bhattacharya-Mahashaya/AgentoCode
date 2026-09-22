import tree_sitter
import tree_sitter_python as tspython
import traceback

print("--- Tree-sitter ---")
try:
    # Try new API
    PY_LANGUAGE = tree_sitter.Language(tspython.language())
    parser = tree_sitter.Parser(PY_LANGUAGE)
    print("Language instantiation worked.")
except Exception as e:
    print("Tree-sitter old instantiation failed:", type(e).__name__, str(e))
    traceback.print_exc()

print("\n--- ChromaDB ---")
import chromadb
from chromadb.config import Settings
try:
    client = chromadb.PersistentClient(path=".chroma_db", settings=Settings(anonymized_telemetry=False))
    print("ChromaDB init worked")
except Exception as e:
    print("ChromaDB init failed:", type(e).__name__, str(e))
    traceback.print_exc()

print("\n--- FastMCP ---")
try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("CodebaseGuardian")
    print("FastMCP init worked")
except Exception as e:
    print("FastMCP init failed:", type(e).__name__, str(e))
    traceback.print_exc()
