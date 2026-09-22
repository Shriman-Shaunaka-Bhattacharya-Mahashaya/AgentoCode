import os
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from pydantic import BaseModel, Field

class FileChunk(BaseModel):
    file_path: str
    content: str
    metadata: Dict[str, Any]

class RAGManager:
    def __init__(self, directory: str, persist_dir: str = ".chroma_db"):
        self.directory = os.path.abspath(directory)
        self.persist_dir = os.path.join(self.directory, persist_dir)
        self.timestamps_file = os.path.join(self.persist_dir, "file_timestamps.json")
        
        # We don't initialize the ChromaDB client immediately to avoid SQLite locks 
        # when running in different processes (e.g., CLI and MCP Server).
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        
        # Setup Tree-sitter for Python
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)
        
        # Ensure persist dir exists
        os.makedirs(self.persist_dir, exist_ok=True)

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy initialization of ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
        return self._client

    @property
    def collection(self):
        """Lazy initialization of ChromaDB collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(name="codebase_chunks")
        return self._collection

    def load_timestamps(self) -> Dict[str, float]:
        if os.path.exists(self.timestamps_file):
            with open(self.timestamps_file, 'r') as f:
                return json.load(f)
        return {}

    def save_timestamps(self, timestamps: Dict[str, float]):
        with open(self.timestamps_file, 'w') as f:
            json.dump(timestamps, f)

    def parse_ast_chunks(self, file_path: str, content: str) -> List[FileChunk]:
        """Parses a Python file into logical chunks (functions/classes) using Tree-sitter."""
        chunks = []
        try:
            tree = self.parser.parse(content.encode('utf-8'))
            root_node = tree.root_node

            def traverse_and_extract(node, current_class=None):
                if node.type == 'class_definition':
                    # Extract class name
                    class_name = None
                    for child in node.children:
                        if child.type == 'identifier':
                            class_name = content[child.start_byte:child.end_byte]
                            break
                    
                    # Store the whole class declaration signature (up to first method or docstring)
                    chunks.append(FileChunk(
                        file_path=file_path,
                        content=f"Class {class_name} defined in {file_path}",
                        metadata={"type": "class_definition", "class_name": class_name, "start_line": node.start_point[0]}
                    ))
                    
                    for child in node.children:
                        if child.type == 'block':
                            traverse_and_extract(child, current_class=class_name)
                
                elif node.type == 'function_definition':
                    func_name = None
                    for child in node.children:
                        if child.type == 'identifier':
                            func_name = content[child.start_byte:child.end_byte]
                            break
                    
                    func_content = content[node.start_byte:node.end_byte]
                    metadata = {"type": "function_definition", "func_name": func_name, "start_line": node.start_point[0]}
                    if current_class:
                        metadata["class_name"] = current_class
                        
                    chunks.append(FileChunk(
                        file_path=file_path,
                        content=func_content,
                        metadata=metadata
                    ))
                else:
                    for child in node.children:
                        traverse_and_extract(child, current_class)

            traverse_and_extract(root_node)
            
            # If no functions/classes found, or it's a small script, add the whole file as a chunk
            if not chunks and content.strip():
                 chunks.append(FileChunk(
                        file_path=file_path,
                        content=content,
                        metadata={"type": "entire_file"}
                    ))
        except Exception as e:
            # Fallback for parsing errors
            print(f"Error parsing AST for {file_path}: {e}")
            chunks.append(FileChunk(
                file_path=file_path,
                content=content,
                metadata={"type": "fallback_entire_file"}
            ))
            
        return chunks

    def purge_file_chunks(self, file_path: str):
        """Deletes all chunks associated with a file path from the vector store."""
        try:
            self.collection.delete(where={"file_path": file_path})
        except Exception as e:
            print(f"Warning: Could not purge old chunks for {file_path}: {e}")

    def index_file(self, file_path: str, timestamps: Dict[str, float]):
        """Indexes a single file if it has changed."""
        if not file_path.endswith('.py'):
            return # Only indexing Python files for now MVP
            
        if not os.path.exists(file_path):
            self.purge_file_chunks(file_path)
            timestamps.pop(file_path, None)
            return

        mtime = os.path.getmtime(file_path)
        if file_path in timestamps and timestamps[file_path] >= mtime:
            return # File hasn't changed
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Delta index: remove old chunks first
            self.purge_file_chunks(file_path)
            
            chunks = self.parse_ast_chunks(file_path, content)
            
            if chunks:
                ids = [f"{file_path}_{i}" for i in range(len(chunks))]
                documents = [c.content for c in chunks]
                metadatas = [c.metadata for c in chunks]
                # Ensure all metadata has file_path for purging later
                for m in metadatas:
                    m["file_path"] = file_path
                
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            timestamps[file_path] = mtime
        except Exception as e:
            print(f"Error indexing {file_path}: {e}")

    def sync_codebase(self):
        """Performs a full pass over the directory to index new/changed files."""
        timestamps = self.load_timestamps()
        
        for root, dirs, files in os.walk(self.directory):
            # Skip hidden dirs like .git, .chroma_db, venv
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'venv']
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.index_file(file_path, timestamps)
                    
        self.save_timestamps(timestamps)

    def search(self, query: str, n_results: int = 5) -> str:
        """Searches the vector store and returns a formatted string of results."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant context found in codebase."
            
        context_parts = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            file_path = meta.get('file_path', 'Unknown file')
            chunk_type = meta.get('type', 'snippet')
            context_parts.append(f"--- File: {file_path} ({chunk_type}) ---\n{doc}\n")
            
        return "\n".join(context_parts)


class CodebaseEventHandler(FileSystemEventHandler):
    def __init__(self, manager: RAGManager):
        self.manager = manager
        
    def _handle_event(self, file_path):
        if not file_path.endswith('.py'):
            return
        # Debounce/throttle logic could go here, but for now just sync immediately
        timestamps = self.manager.load_timestamps()
        self.manager.index_file(file_path, timestamps)
        self.manager.save_timestamps(timestamps)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
             # Purge deleted file
             timestamps = self.manager.load_timestamps()
             self.manager.purge_file_chunks(event.src_path)
             timestamps.pop(event.src_path, None)
             self.manager.save_timestamps(timestamps)

def start_watchdog(directory: str) -> Observer:
    manager = RAGManager(directory)
    # Perform initial sync
    manager.sync_codebase()
    
    event_handler = CodebaseEventHandler(manager)
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    return observer
