# 🛡️ AgentoCode

AgentoCode is a fully autonomous, CLI-based AI agent designed to act as your local codebase guardian and senior developer. It does not just chat about your code; it possesses full contextual memory, semantically understands your file structure, natively edits files, and executes terminal commands to test its own fixes within a secure sandbox.

Built by marrying local **Retrieval-Augmented Generation (RAG)** for memory with the **Model Context Protocol (MCP)** for execution, and wrapped in a beautiful terminal interface.

---

## ✨ Key Features

- **Semantic AST Chunking:** Uses `tree-sitter` to parse Python files by logical boundaries (Classes & Methods) rather than arbitrary character limits, tagging each chunk with rich metadata.
- **Real-Time Watchdog Sync:** A background daemon silently monitors your selected workspace. If you (or the agent) edit a file, the vector database is instantly updated in real-time.
- **Conversational Memory:** The ReAct agent retains full chronological context of your conversation across multiple prompts and tool calls.
- **Directory Sandboxing:** The backend MCP server strictly enforces file access boundaries, preventing the LLM from reading, writing, or executing commands outside of the approved sandbox directory.
- **Self-Healing LLM Loop:** If the LLM hallucinates a bad tool call or encounters an error, the agent catches the crash and feeds the error back to the LLM so it can learn and correct its mistake.
- **Rich Interactive CLI:** A high-performance terminal UI using `prompt_toolkit` and `rich` for markdown rendering, syntax highlighting, and persistent chat history.

---

## 🛠️ Tech Stack & Components

AgentoCode strictly decouples infrastructure into distinct layers to maintain security and performance:

### 1. `cli.py` (The Interface)
The entry point of the application. It replaces traditional web UIs with a fast, async terminal interface.
- **Tech Used:** `prompt_toolkit` (for async user input and persistent `.cg_history`), `rich` (for markdown formatting and status spinners).
- **Function:** Initializes the background watchdog, sets up the agent configuration (including the sandbox directory), and runs the chat loop.

### 2. `agent.py` (The Brain)
The reasoning engine running an asynchronous ReAct (Reason + Act) loop.
- **Tech Used:** `groq` (AsyncGroq Client), `mcp` (ClientSession).
- **Model:** `openai/gpt-oss-120b` via Groq for ultra-fast, intelligent tool usage with massive context and output budgets.
- **Function:** Dynamically spawns the MCP server as a subprocess, retrieves available tools, maintains conversational memory, and orchestrates tool execution. If a tool fails, it captures the error and feeds it back into the loop.

### 3. `server.py` (The Hands)
The execution layer exposing local OS capabilities via the Model Context Protocol (MCP).
- **Tech Used:** `mcp` (MCPServer).
- **Function:** Exposes `read_file`, `write_file`, `run_command`, and `rag_search`. It intercepts every file request to ensure the absolute path strictly resides within the `--allowed-dir` sandbox.

### 4. `rag.py` (The Memory)
The vector database and file syncing engine.
- **Tech Used:** `chromadb` (Vector storage), `tree-sitter` & `tree-sitter-python` (AST Parsing), `watchdog` (File monitoring).
- **Function:** Scans the sandbox directory, parses Python files into logical AST chunks (classes/functions), and stores them in ChromaDB. The `watchdog` daemon runs in the background to automatically re-index modified files.

---

## ⚙️ Prerequisites

* **Python 3.10+** (Developed on Python 3.14.x)
* A free [Groq API Key](https://console.groq.com/) for lightning-fast inference.

---

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YourUsername/AgentoCode.git
cd AgentoCode
```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install the required dependencies:**
All dependencies are strictly pinned in `requirements.txt`.
```bash
pip install -r requirements.txt
```

4. **Configure the environment:**
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

---

## 💻 How to Run & Use

1. **Launch the CLI:**
Start the agent by providing the directory you want it to guard. This directory becomes its strictly enforced sandbox.
```bash
python cli.py --dir ./sample_project
```
> **Note on First-Time Indexing:** The very first time you run the agent, `chromadb` will download the default embedding model (`all-MiniLM-L6-v2`, ~80MB) in the background. Depending on your network speed, this may make the initial indexing phase appear frozen or take several minutes. Once cached, subsequent runs and real-time syncing will be instantaneous.

*(The background Watchdog will immediately index the folder into a hidden `.chroma_db` database and listen for real-time changes).*

2. **Interact with the Agent:**
Type your queries directly into the interactive prompt. Do not treat this like a standard ChatGPT prompt. Give it actionable tasks! 
- *"How does the `SimpleCalculator` class handle division errors? Rewrite the method to return 0 instead of raising an error."*
- *"Search the codebase for the RAG indexing logic. Read the file, change the chunk size from 4000 to 2500, save it, and then run `python rag.py` to test."*

3. **CLI Commands:**
- `/exit` or `/quit` : Safely shutdown the background watchdog and exit the CLI.
- `/clear` : Clear the agent's conversational memory (useful if the context window gets too large).
- `/sync` : Reminder that memory is synced automatically on file save.

---

## ⚠️ Security Notice

This agent has access to a terminal execution tool (`run_command`) and file overwrite tools (`write_file`). 
While it features path-based sandboxing to restrict access to the directory you explicitly select via `--dir`, it is still executing terminal commands on your local OS. **Do not select root directories or critical system folders as your sandbox.** Use it strictly within isolated project workspaces.
