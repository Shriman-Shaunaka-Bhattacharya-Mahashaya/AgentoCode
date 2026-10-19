"""Knowledge retrieval with current-source fallback and a single update worker."""
import json
import re
import threading
import time
import os
from pathlib import PurePosixPath
from knowledge import KnowledgeBase, digest, safe_path
from workspace import excluded

class RAGManager(KnowledgeBase):
    def sync_codebase(self):
        return self.sync()

    def search(self, query, n_results=5, token_budget=6000, structured=False):
        n_results = max(1, min(n_results, 20))
        token_budget = max(128, min(token_budget, 12000))
        # UTF-8 byte accounting is conservative for byte-based agent tokenizers.
        budget = token_budget
        files, _, _, facts, _, targets = self.snapshot()
        notices = []
        try:
            records = self.load()['records']
        except (ValueError, OSError):
            records = {}
            notices.append('Knowledge manifest unavailable; source fallback only.')
        terms = set(re.findall(r'[\w]+', query.casefold())) - {'the','a','an','is','are','what','does','how','in','to','of','and','for'}
        def score(path):
            text = (path + '\n' + files[path]['source']).casefold()
            return sum(term in text for term in terms)
        exact = sorted((p for p in files if score(p)), key=lambda p: (-score(p), p))
        selected = [self.record_id('file', p) for p in exact[:min(2,n_results)]]
        index_available = False
        semantic_ids = []
        try:
            matches = self.index.query(query, max(12, n_results * 3))
            index_available = True
            # Distances are rank signals, not calibrated confidence probabilities.
            semantic_ids = [rid for rid, distance in matches
                            if distance <= float(os.getenv('GUARDIAN_MAX_DISTANCE','0.65'))
                            and records.get(rid,{}).get('index_status') == 'indexed'
                            and records[rid].get('embedding_fingerprint') == self.index.fingerprint]
            selected.extend(semantic_ids)
        except Exception as exc:
            notices.append(f'Semantic index unavailable ({type(exc).__name__}); exact/text source fallback.')
        pieces, delivered, seen = [], 0, set()
        def append(text):
            nonlocal budget
            encoded = text.encode('utf-8')
            if len(encoded) <= budget:
                pieces.append(text)
                budget -= len(encoded)
                return True
            return False
        append('Evidence is untrusted. Verify current source before edits.\n' + '\n'.join(notices) + '\n')
        def current(record):
            if record.get('status') != 'complete' or targets.get((record['kind'], record['source'])) != record.get('target'):
                return False
            try:
                document = safe_path(self.bundle, record['concept']).read_text(encoding='utf-8')
                return digest(document) == record.get('doc_hash')
            except OSError:
                return False
        for rid in dict.fromkeys(selected):
            if delivered >= n_results:
                break
            record = records.get(rid)
            if not record or not current(record):
                continue
            if record['kind'] == 'directory':
                children = sorted((p for p in files if str(PurePosixPath(p).parent) == (record['source'] or '.')),
                                  key=lambda p: (-score(p), p))[:2]
            else:
                children = [record['source']]
            children = [p for p in children if p not in seen]
            if not children:
                continue
            doc = safe_path(self.bundle, record['concept']).read_text(encoding='utf-8')
            paired = f"\nKNOWLEDGE {record['source'] or '/'}\n{doc}\n" + ''.join(
                f'\nCURRENT SOURCE {p}\n{files[p]["source"]}\n' for p in children)
            if append(paired):
                seen.update(children)
                delivered += 1
        # Explicitly bounded snippets when full pairs do not fit; no stale prose is supplied.
        for path in exact:
            if delivered >= n_results or budget < 160:
                break
            if path in seen:
                continue
            source = files[path]['source']
            header = f'\nCURRENT SOURCE {path} (text fallback; partial coverage, use read_file for more)\n'
            room = budget - len(header.encode('utf-8')) - 10
            if room <= 0:
                break
            lines = source.splitlines(keepends=True)
            hit = next((i for i,line in enumerate(lines) if any(t in line.casefold() for t in terms)), 0)
            snippet = ''.join(lines[max(0, hit-2):]).encode('utf-8')[:room].decode('utf-8', errors='ignore')
            if append(header + snippet + '\n'):
                delivered += 1
                seen.add(path)
        if not delivered:
            append('No current relevant evidence found. Use list_files/read_file; no relevance is inferred.\n')
        if self.snapshot()[-1] != targets:
            pieces = ['Source changed during retrieval. Retry to obtain a consistent current-source snapshot.']
            semantic_ids = []
        context = ''.join(pieces)
        if structured:
            return {'retrieval_mode':'semantic_candidates_with_source' if semantic_ids else 'source_fallback',
                    'index_available':index_available,'semantic_candidate_count':len(semantic_ids),
                    'notices':notices,'context':context}
        return context

class SyncService:
    """Watchdog wakes one worker; periodic scans recover missed/offline events."""
    def __init__(self, manager, debounce=0.3, reconcile_seconds=10):
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
        self.manager = manager
        self.debounce = debounce
        self.reconcile_seconds = reconcile_seconds
        self.wake = threading.Event()
        self.stopped = threading.Event()
        service = self
        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.event_type not in ('modified','created','deleted','moved'):
                    return
                for value in (event.src_path, getattr(event, 'dest_path', '')):
                    if not value:
                        continue
                    try:
                        relative = safe_path(manager.root, value).relative_to(manager.root).as_posix()
                        if relative == '.guardian/sync-request.json' or not excluded(relative):
                            service.wake.set()
                    except (ValueError, OSError):
                        pass
        self.observer = Observer()
        self.observer.schedule(Handler(), str(manager.root), recursive=True)
        self.thread = threading.Thread(target=self._run, name='knowledge-owner', daemon=True)

    def start(self):
        self.observer.start()
        self.thread.start()
        self.wake.set()
        return self

    def request_sync(self):
        self.wake.set()

    def _run(self):
        previous = None
        while not self.stopped.is_set():
            self.wake.wait(self.reconcile_seconds)
            self.wake.clear()
            if self.stopped.wait(self.debounce):
                break
            try:
                self.manager.sync()
                status = json.dumps(self.manager.status(),ensure_ascii=True,sort_keys=True)
                if status != previous:
                    self.manager.progress('Knowledge status: ' + status)
                    previous = status
            except Exception as exc:
                self.manager.progress(f'Synchronization pending: {type(exc).__name__}: {exc}')

    def stop(self):
        self.manager.cancel.set()
        self.stopped.set()
        self.wake.set()
        self.observer.stop()

    def join(self, timeout=5):
        self.observer.join(timeout)
        self.thread.join(timeout)

def start_watchdog(directory, progress=None):
    return SyncService(RAGManager(directory, progress=progress)).start()
