# AgentoCode

AgentoCode is an interactive Python agent for a selected project. A coordinator can delegate independent subtasks to multiple worker agents and combine their findings. Groq supplies the agent models and independent knowledge explanations. **Sentence Transformers runs the embedding model locally inside the persistent MCP server**, and Chroma stores one vector per completed knowledge concept.

Validation on Windows includes **130 passing tests** (75.76 seconds, no failures or skips) under the CI test environment, plus earlier live CLI testing with Groq and Nomic on CPU. Earlier combined automated/live coverage measured **95% of statements, 87% of branches, 93% overall**. Live checks covered concurrent editing workers, rate-limit recovery, source-driven vector replacement, deletion, cached restart and MCP stdout isolation. These results establish tested behavior, not general retrieval accuracy. Detailed run reports are local artifacts excluded from the repository.

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

This Windows bootstrap uses `curl.exe`, checks the upstream model-weight checksum, and writes a local asset manifest. A cache lock prevents overlapping downloads; verified temporary weights replace the cached file atomically. Rerunning the bootstrap repairs corrupt weights, including corruption that leaves the file size unchanged. Completed model assets are subsequently checked before loading. Without a local snapshot, the adapter can download the pinned Hub revision; `GUARDIAN_LOCAL_FILES_ONLY=1` disables that fallback.

CPU is the default and was used in live testing. The installed test environment's PyTorch build has no CUDA support. GPU use requires a compatible CUDA-enabled PyTorch installation and explicit `GUARDIAN_EMBED_DEVICE=cuda`; GPU fit/performance have not been verified.

## Run the CLI

```powershell
python cli.py --dir ./sample_project
```

Run the interactive CLI in a terminal. Startup launches one persistent MCP process and queues knowledge synchronization. Model loading/indexing happens in its background worker. Use `/status` to inspect progress; interaction can use current source while an index is pending or unavailable.

| Command | Behavior |
| --- | --- |
| `/status` | Show fresh concept/index counts, stale records, index verification, failures, model state/device/load count and embedding-call counters. |
| `/agents` | Show the last worker batch, results, states, usage, confirmed/unconfirmed writes and provider retries. |
| `/sync` | Queue reconciliation and retries. |
| `/clear` | Clear CLI conversational history. |
| `/exit`, `/quit`, EOF | Close the persistent connection and stop the server/worker. |

Example requests:

- “Find where past calculations are remembered, verify the source, and explain whether history is copied.”
- “What happens when the calculator receives a zero denominator?”
- “Add a documented absolute-value method to this selected sample project.”

The agent can inspect and write selected workspace files. It cannot run shell commands; execute tests yourself in your chosen environment.

## Coordinator and worker agents

The conversational agent coordinates the task. Its `delegate_tasks` tool submits a batch of smaller independent tasks, waits for their results and lets the coordinator verify/integrate them. `worker_status` retrieves the last batch so completed findings can be reused after a provider error. Workers have fresh contexts containing their assigned instructions, the selected workspace and role-specific tools. They cannot start more workers.

By default, up to **3 workers** run concurrently. Up to 6 tasks fit in one batch, with at most 3 batches and 12 total tasks per user turn. Each worker has a 120-second time limit, at most 6 model rounds, the normal input budget and a bounded result size. A worker completion state means it returned a summary; the coordinator must still verify the findings against source.

```powershell
# Limit concurrency to two workers
python cli.py --dir ./sample_project --workers 2

# Disable delegation for a session
python cli.py --dir ./sample_project --single-agent
```

Example: “Use two read-only workers: inspect calculator.py for history behavior and main.py for error handling. Combine the findings and report each worker's status.”

Workers default to **read-only**. When edits are requested, the coordinator must assign exact `writable_paths` to each editing worker. Overlapping assignments and directory/file collisions are rejected before a batch starts. Workers cannot write outside their assignment, call coordinator tools, or bypass workspace checks. Disjoint writes use per-file locks and coordinator-owned snapshot hashes; a changed source produces a conflict instead of overwriting that observed revision. Each successful write triggers the normal knowledge refresh. Schedule dependent edits in later batches.

All workers share the existing MCP connection, knowledge worker and resident embedding model. Worker failures and timeouts are reported independently. Cancellation stops pending workers; completed writes persist, and interrupted write attempts are marked for inspection rather than rolled back. External processes can still race filesystem changes; these guards do not provide OS-level transactions.

Each worker adds Groq requests and tokens. Shared bounded backoff honors rate-limit cooldowns, retaining completed tool results while retrying. A request stops after five attempts or a 60-second waiting budget; worker timeouts still apply. `/agents` shows retries and per-worker usage. `--worker-model` can select a separate Groq worker model; otherwise workers use the coordinator model.

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

Atomic publication and manifest reads retry transient Windows sharing/access failures with bounded backoff. Persistent failures remain visible in synchronization status and diagnostics. Status excludes stale source/document revisions from current counts; `index_verified` indicates a verified active embedding fingerprint, rather than proving retrieval quality.

Corrupt generation-cache JSON is regenerated, and truncated model responses cannot be published as complete explanations. Same-line AST statements are included once. The revised generation and vector-normalization fingerprints cause incompatible cached explanations/vectors to be rebuilt automatically.

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

The server exposes `list_files`, `read_file`, `write_file`, `rag_search`, `read_concept`, `knowledge_status` and `sync_knowledge`. Delegation and worker-status tools run in the coordinator client, with worker workspace calls sent through the same server. One resident embedding model serves synchronization and queries across CLI turns, with serialized inference and batch size one.

**MCP stdout is reserved exclusively for JSON-RPC.** Before application/library imports, the server duplicates its protocol pipe and redirects ordinary Python/native stdout to stderr. The transport writes through the private pipe. Hugging Face progress is disabled; model diagnostics/errors go to the selected workspace's `.guardian/mcp-server.log`, not the CLI prompt or protocol stream. A raw-wire regression injects both Python and native stdout noise during a model-load failure and confirms protocol recovery. A real-model offline protocol probe also passed.

`rag_search` returns structured `retrieval_mode`, `index_available`, candidate count, notices and bounded knowledge/current-source context. A nonempty fallback response is not evidence of functioning vector retrieval. File/folder hits are checked for freshness and sources are deduplicated; folders expand to at most two immediate files. Exact/text fallback and bounded read tools cover unavailable indexing or oversized context.

Resolved-path containment rejects outside-workspace paths, links/junctions and hidden/generated tool paths. Windows device names, alternate streams and trailing-dot/space aliases are rejected before file access. The agent application root and ancestors cannot be selected as its workspace; the bundled sample project can. These file boundaries are not an OS sandbox against a separate process racing filesystem changes. The model has no arbitrary execution tool.

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

### Automatic checks on GitHub

[Tests workflow](.github/workflows/tests.yml) runs on every push and pull request, and can be started manually from the Actions tab. It targets Windows with Python 3.12, installs the pinned development dependencies, runs `pip check` and the entire regression/integration suite, and requires at least **90% combined statement/branch coverage**. Failed tests or insufficient coverage fail the job. Superseded runs on the same branch are cancelled.

No Groq secret or embedding snapshot is needed for CI. Tests use deterministic model fixtures with real Chroma/MCP/Watchdog integrations. Dotenv loading and Hub downloads are disabled during tests; dependency installation still needs network access. Fresh-process CLI checks verify help, missing credentials and refusal to select the application's own workspace. An additional real-Chroma integration checks no-op reuse, stale-source refresh, restart, deletion and recreation under stable IDs.

The local CI-equivalent run passed all **130 tests** with **92.78% overall coverage**, exceeding the 90% threshold. A hosted Actions run still needs to execute after this workflow is pushed.

Each run retains `junit.xml` and `coverage.xml` in the `windows-python312-test-reports` artifact for 14 days, including available reports after a test failure. Application caches, logs and credentials are excluded from the upload. Reports are also ignored locally under `test-results/`.

Checks run after GitHub receives a push. To prevent merging failing changes, configure the repository's branch protection/ruleset to require **Windows / Python 3.12**. That repository setting must be enabled separately. See [GitHub's Python CI guide](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) for the Actions interface. Live Groq/Nomic evaluations remain separate from these deterministic checks.

### Local coverage and validation limits

For statement/branch coverage, including Python subprocesses:

```powershell
python -m coverage erase
python -m coverage run -m pytest -q
python -m coverage combine
python -m coverage report
```

The measured coverage above combines the full suite with live CLI and cached-model protocol/input probes; the suite alone may produce a different percentage. Tests use deterministic model fixtures where appropriate and real Chroma/MCP/Watchdog integrations. They cover concurrent scheduling, context/tool isolation, shared budgets, ownership/conflicts, timeout/cancellation and uncertain writes, malformed rate-limit headers, knowledge lifecycle, parser/cache failures, download corruption/resume, embedding boundaries, workspace containment and raw stdout isolation.

Live Groq tests used an isolated copy of `sample_project`. Two editing workers created separate helpers, and eight arithmetic checks passed after reviewing their exact source. External edits updated only affected file/parent vectors under stable IDs; deletion returned the index from six to four concepts. Cached restart changed no document bytes/timestamps and made no generation/embedding calls. A real offline MCP probe produced four valid JSON-RPC frames with no stray stdout. Real CPU inference accepted exactly 2,048 rendered tokens and rejected 2,049 before encoding.

Retrieval remains a limitation: six hand-written semantic-only queries over four sample concepts placed the expected file first in **3/6 cases**, within two in **3/6**, with MRR **0.653**. Folder/overview concepts sometimes outranked the target file. These are small diagnostic results, not a held-out benchmark or a measurement of combined text/vector retrieval. The distance cutoff remains uncalibrated; verify current source before edits.

Testing used an existing Python 3.12.10 environment. `pip check` and a fresh wheel-only dependency resolver dry run passed; a separate clean-machine installation was not performed. Other operating systems, Qwen, GPU inference, extended context, multilingual retrieval, large-project performance and peak memory remain unverified.

## Repository contents

| Files | Purpose |
| --- | --- |
| `cli.py`, `agent.py`, `delegation.py`, `server.py` | Interactive CLI, coordinator/worker tool loops, bounded scheduling and persistent MCP server. |
| `workspace.py`, `facts.py` | Workspace eligibility, file boundaries and Python structural facts. |
| `generation.py`, `knowledge.py` | Independent explanations, knowledge documents and incremental lifecycle. |
| `embeddings.py`, `rag.py` | Local model loading, Chroma retrieval and background synchronization. |
| `download_model.py` | Download and verify the pinned local Nomic snapshot. |
| `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.coveragerc`, `tests/` | Runtime/development dependencies, subprocess coverage configuration and maintained regression tests. |
| `sample_project/` | Calculator example for interactive testing. |

Generated knowledge, model caches, databases, logs, local run reports, exploratory scripts and virtual environments are ignored. Application source, sample source and maintained tests belong in version control. Environment examples may be committed as `.env.example` or `.env.template` with placeholder values only.
