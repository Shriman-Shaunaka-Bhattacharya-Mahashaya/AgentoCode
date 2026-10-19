# AgentoCode

AgentoCode is an interactive Python agent for a selected project. Groq supplies the conversational model and independent knowledge explanations. **Sentence Transformers runs the embedding model locally inside the persistent MCP server**, and Chroma stores one vector per completed knowledge concept.

Validation includes **58 passing regression tests** and live CLI testing with Groq and Nomic on CPU. Live checks covered semantic search, stable-ID vector updates, deletion, cached restart and MCP stdout isolation. Detailed run reports are local artifacts excluded from the repository.

## Setup on Windows

Python 3.12.10 was tested. Create an environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in your environment or the project's local `.env` file. Selected source is sent to Groq for explanations and agent reasoning; embedding inference runs locally. Credentials and generated state are ignored by Git.

Download the pinned Nomic snapshot into the ignored project cache:

```powershell
python download_model.py
```

This Windows bootstrap uses `curl.exe`, checks the upstream model-weight checksum, and writes a local asset manifest. It downloads the selected model and configuration files rather than a separate model server. Completed model assets are subsequently checked before loading. Without a local snapshot, the adapter can download the pinned Hub revision; `GUARDIAN_LOCAL_FILES_ONLY=1` disables that fallback.

CPU is the default and was used in live testing. The installed test environment's PyTorch build has no CUDA support. GPU use requires a compatible CUDA-enabled PyTorch installation and explicit `GUARDIAN_EMBED_DEVICE=cuda`; GPU fit/performance have not been verified.

## Run the CLI

```powershell
python cli.py --dir ./sample_project
```

Startup launches one persistent MCP process and queues knowledge synchronization. Model loading/indexing happens in its background worker. Use `/status` to inspect progress; interaction can use current source while an index is pending or unavailable.

| Command | Behavior |
| --- | --- |
| `/status` | Show concept/index counts, failures, model state/device/load count and embedding-call counters. |
| `/sync` | Queue reconciliation and retries. |
| `/clear` | Clear CLI conversational history. |
| `/exit`, `/quit`, EOF | Close the persistent connection and stop the server/worker. |

Example requests:

- “Find where past calculations are remembered, verify the source, and explain whether history is copied.”
- “What happens when the calculator receives a zero denominator?”
- “Add a documented absolute-value method to this selected sample project.”

The agent can inspect and write selected workspace files. It cannot run shell commands; execute tests yourself in your chosen environment.

## Knowledge and synchronization

The selected workspace owns its exact `knowledge_base/` directory. Ownership/layout must match before reuse. Documents follow OKF v0.2 frontmatter and application-specific identity/fingerprint/status fields.

```text
sample_project/
  calculator.py
  knowledge_base/
    index.md
    files/calculator.py/content.md
    directories/content.md
    directories/index.md
  .guardian/
    vectors/
    mcp-server.log
```

Readable source paths/extensions are preserved in separate file/directory namespaces. Reserved names and the escape marker use reversible encoding. Generated state, VCS/hidden paths, dependency environments, ignored files, credential names/extensions, binaries and files larger than 512,000 bytes are excluded from source generation.

Each eligible file receives independently generated explanations and separately rendered Python AST facts. Unknown/dynamic call bindings remain unresolved. Unsupported text formats have explicit structural limitations. Directory concepts aggregate immediate children bottom up. Oversized Python input is split at top-level AST boundaries; oversized individual units use bounded fragments. Generation has retries, cancellation and a completed-unit cache.

User saves, agent writes and external edits follow one hash-based lifecycle. The worker refreshes affected files, imported-file dependents and parents, then recomputes embeddings under stable concept IDs. Additions receive new IDs. Duplicate/no-op notifications make no new generation/embedding calls. Source/dependency changes re-embed even if generated prose is identical.

Publication and vector replacement are recoverable stages. Retrieval rejects stale evidence using current hashes. Moves/deletions clean local concepts immediately and queue vector deletion independently of model availability. Manual knowledge-body edits are validated and reindexed. Corrupt indexes with recognized database-corruption errors rebuild into a fresh store while preserving old data; malformed manifests remain preserved for diagnosis and source fallback.

Atomic publication retries transient Windows sharing/access failures with bounded backoff. Persistent failures remain visible in synchronization status and diagnostics.

## Embedding profiles

| Profile | Model | Full width | Status |
| --- | --- | --- | --- |
| `nomic_v15_st` | `nomic-ai/nomic-embed-text-v1.5`, revision `e9b6763023c676ca8431644204f50c2b100d9aab` | 768 | Default; live CPU inference/indexing verified. |
| `qwen4b_st` | `Qwen/Qwen3-Embedding-4B`, explicit pinned revision required | 2,560 | Optional compatible checkpoint; not downloaded, fit-tested or benchmarked. |

Nomic documents use `search_document: ` and questions use `search_query: `. Qwen uses its retrieval instruction on queries and native last-token pooling. The adapter checks loaded pooling and width, applies prompts exactly once and normalizes floating output vectors.

The verified operational cap is **2,048 tokens**. Preflight tokenizes the exact rendered input including special tokens with truncation disabled. Overlength concepts stay intact and visibly unindexed; source fallback remains available. The advertised longer model context is not automatically enabled. No silent model switch, dimension cropping or window averaging occurs.

The fingerprint includes model assets/revision, runtime versions, device/dtype, pooling, templates, dimensions, normalization, context and serialization. The new runtime uses its own Chroma collection even when dimensions match older data. Existing knowledge is reused during re-embedding; legacy data is preserved.

Qwen's old GGUF Q4_K_M serving configuration is superseded. A compatible checkpoint or separately validated quantization configuration is required; a small quantized file does not guarantee a small Sentence Transformers runtime footprint.

## MCP, output and retrieval

The server exposes `list_files`, `read_file`, `write_file`, `rag_search`, `read_concept`, `knowledge_status` and `sync_knowledge`. One resident model serves synchronization and queries across CLI turns, with serialized inference and batch size one.

**MCP stdout is reserved exclusively for JSON-RPC.** Before application/library imports, the server duplicates its protocol pipe and redirects ordinary Python/native stdout to stderr. The transport writes through the private pipe. Hugging Face progress is disabled; model diagnostics/errors go to the selected workspace's `.guardian/mcp-server.log`, not the CLI prompt or protocol stream. A raw-wire regression injects both Python and native stdout noise during a model-load failure and confirms protocol recovery. A real-model offline protocol probe also passed.

`rag_search` returns structured `retrieval_mode`, `index_available`, candidate count, notices and bounded knowledge/current-source context. A nonempty fallback response is not evidence of functioning vector retrieval. File/folder hits are checked for freshness and sources are deduplicated; folders expand to at most two immediate files. Exact/text fallback and bounded read tools cover unavailable indexing or oversized context.

Resolved-path containment rejects outside-workspace paths, links/junctions and hidden/generated tool paths. The agent application root and ancestors cannot be selected as its workspace; the bundled sample project can. These file boundaries are not an OS sandbox against a separate process racing filesystem changes. The model has no arbitrary execution tool.

## Configuration

| Environment variable | Default / meaning |
| --- | --- |
| `GROQ_API_KEY` | Required for CLI model calls. |
| `GUARDIAN_GENERATION_MODEL` | `openai/gpt-oss-120b` |
| `GUARDIAN_EMBED_PROFILE` | `nomic_v15_st` |
| `GUARDIAN_EMBED_DEVICE` | `cpu`; explicit `cuda` requires a compatible environment. |
| `GUARDIAN_QWEN_REVISION` | Required 40-character commit for optional `qwen4b_st`. |
| `GUARDIAN_LOCAL_FILES_ONLY` | `0`; set `1` to prohibit fallback Hub downloads when the pinned local snapshot is absent. |
| `GUARDIAN_MAX_DISTANCE` | `0.65`; uncalibrated cosine-distance heuristic, not a confidence probability. |

## Tests and limitations

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python sample_project/main.py
```

Tests cover lifecycle/failure recovery, direct-model contracts, real Chroma, persistent tool transport, watcher updates, workspace containment and raw stdout isolation. Live tests additionally verify real Nomic vectors, stable-ID replacement, unchanged siblings, deletion and cached restart.

The live checks cover a small sample project. Qwen, GPU inference, extended context, multilingual retrieval, held-out recall/MRR and peak memory measurements remain unverified.

## Repository contents

| Files | Purpose |
| --- | --- |
| `cli.py`, `agent.py`, `server.py` | Interactive CLI, conversational tool loop and persistent MCP server. |
| `workspace.py`, `facts.py` | Workspace eligibility, file boundaries and Python structural facts. |
| `generation.py`, `knowledge.py` | Independent explanations, knowledge documents and incremental lifecycle. |
| `embeddings.py`, `rag.py` | Local model loading, Chroma retrieval and background synchronization. |
| `download_model.py` | Download and verify the pinned local Nomic snapshot. |
| `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `tests/` | Runtime/development dependencies and maintained regression tests. |
| `sample_project/` | Calculator example for interactive testing. |

Generated knowledge, model caches, databases, logs, local run reports, exploratory scripts and virtual environments are ignored. Application source, sample source and maintained tests belong in version control. Environment examples may be committed as `.env.example` or `.env.template` with placeholder values only.
