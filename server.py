"""Workspace-bound MCP tools. Arbitrary commands are deliberately not exposed."""
import argparse
import json
import os
import sys
import io
import asyncio
from contextlib import asynccontextmanager

def reserve_protocol_stream():
    """Keep MCP's pipe private. Python/native stdout diagnostics go to stderr."""
    protocol = io.TextIOWrapper(os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0),
                              encoding='utf-8',write_through=True)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    sys.stdout = sys.stderr
    for name,value in {'HF_HUB_DISABLE_PROGRESS_BARS':'1','TQDM_DISABLE':'1',
                       'TOKENIZERS_PARALLELISM':'false','TRANSFORMERS_VERBOSITY':'error'}.items():
        os.environ[name] = value
    return protocol

# This runs before importing libraries that might print during import/model loading.
_protocol_stdout = reserve_protocol_stream() if __name__ == '__main__' else None
from pathlib import Path
from workspace import safe_path, validate_agent_workspace, inventory, excluded
from knowledge import atomic_write, digest
from rag import RAGManager, SyncService

class WorkspaceTools:
    def __init__(self, directory, manager=None):
        self.root = validate_agent_workspace(directory)
        self.manager = manager or RAGManager(self.root,
            progress=lambda message:print(message,file=sys.stderr,flush=True))
        self.service = None

    def path(self, relative):
        path = safe_path(self.root, relative)
        if excluded(path.relative_to(self.root)):
            raise ValueError('Generated, hidden, or credential paths are not accessible to agent tools')
        return path

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> str:
        """Read current source, bounded to 200 lines and 16000 UTF-8 bytes."""
        value = self.path(path)
        if value.stat().st_size > 512_000:
            raise ValueError('File exceeds read limit')
        text = value.read_text(encoding='utf-8')
        if '\0' in text:
            raise ValueError('Binary input')
        start_line = max(1, start_line)
        lines = text.splitlines()[start_line-1:start_line-1+max(1,min(max_lines,200))]
        return '\n'.join(f'{i}: {line}' for i,line in enumerate(lines,start_line)).encode('utf-8')[:16000].decode('utf-8',errors='ignore')

    def write_file(self, path: str, content: str) -> str:
        """Replace a source file inside the selected workspace and queue knowledge refresh."""
        if len(content.encode('utf-8')) > 512_000:
            raise ValueError('File exceeds write limit')
        target = self.path(path)
        atomic_write(target, content)
        # Persist notification for the owning worker; source hash checks invalidate evidence immediately.
        atomic_write(self.manager.state / 'sync-request.json', json.dumps({'path': str(target.relative_to(self.root))}))
        if self.service:
            self.service.request_sync()
        return f'Wrote {path}; knowledge refresh queued. Current-source validation excludes older knowledge.'

    def list_files(self) -> str:
        """List eligible workspace files for exact-path/source fallback."""
        files, _, skipped = inventory(self.root)
        return json.dumps({'files': list(files)[:300], 'omitted': max(0,len(files)-300), 'excluded_count': len(skipped)})

    def rag_search(self, query: str) -> str:
        """Retrieve knowledge concepts plus current associated source, with bounded text fallback."""
        return json.dumps(self.manager.search(query,structured=True),ensure_ascii=False)

    def knowledge_status(self) -> str:
        """Report document generation/indexing status and failures."""
        return json.dumps(self.manager.status())

    def sync_knowledge(self) -> str:
        """Queue knowledge reconciliation with the resident synchronization worker."""
        if not self.service:
            return 'No background worker in this server; start the CLI or server with --watch.'
        self.service.request_sync()
        return 'Knowledge reconciliation queued.'

    def read_concept(self, source_path: str, kind: str = 'file') -> str:
        """Read a fresh concept and its related concept links for a source identity."""
        if kind not in ('file', 'directory'):
            raise ValueError('kind must be file or directory')
        self.path(source_path or '.')
        record = self.manager.load()['records'].get(self.manager.record_id(kind,source_path))
        targets = self.manager.snapshot()[-1]
        if not record or record.get('status') != 'complete' or targets.get((kind,source_path)) != record.get('target'):
            return 'Knowledge is absent or stale; read current source instead.'
        document = safe_path(self.manager.bundle,record['concept']).read_text(encoding='utf-8')
        if digest(document) != record.get('doc_hash'):
            return 'Knowledge has an unvalidated edit; synchronization is required.'
        raw = document.encode('utf-8')
        return raw[:15900].decode('utf-8',errors='ignore') + ('\n[Partial coverage: document exceeds read budget.]' if len(raw)>15900 else '')

def create_server(directory, watch=False):
    from mcp.server.mcpserver import MCPServer
    workspace = WorkspaceTools(directory)
    @asynccontextmanager
    async def lifespan(server):
        if watch:
            workspace.service = SyncService(workspace.manager).start()
        try:
            yield {}
        finally:
            if workspace.service:
                workspace.service.stop()
                await asyncio.to_thread(workspace.service.join,5)
    mcp = MCPServer('CodebaseGuardian',lifespan=lifespan)
    for name in ('read_file','write_file','list_files','rag_search','knowledge_status','read_concept','sync_knowledge'):
        mcp.tool()(getattr(workspace,name))
    return mcp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allowed-dir', required=True)
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent/'.env')
    async def serve():
        import anyio
        from mcp.server.stdio import stdio_server
        mcp = create_server(args.allowed_dir,watch=args.watch)
        async with stdio_server(stdout=anyio.wrap_file(_protocol_stdout)) as (read,write):
            await mcp._lowlevel_server.run(read,write,mcp._lowlevel_server.create_initialization_options())
    asyncio.run(serve())
