"""Owned OKF v0.2 bundles, incremental generation, and recoverable indexing."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote
import json
import os
import threading
import time
import uuid
import yaml
from workspace import digest, inventory, safe_path
from facts import extract, resolve
from generation import GroqGenerator
from embeddings import ChromaIndex

LAYOUT = 1

def concept_path(kind, source):
    if kind not in ('file', 'directory') or PurePosixPath(source).is_absolute() or '..' in PurePosixPath(source).parts:
        raise ValueError('Invalid concept identity')
    # Preserve readable source paths/extensions; escape only reserved names and the escape marker.
    parts = [('~' + component.encode('utf-8').hex()) if component.casefold() in
             {'content.md','index.md','log.md'} or component.startswith('~') else component
             for component in PurePosixPath(source).parts]
    return '/'.join([('files' if kind == 'file' else 'directories'), *parts, 'content.md'])

def source_path(concept):
    parts = PurePosixPath(concept).parts
    if parts[0] not in ('files', 'directories') or parts[-1] != 'content.md':
        raise ValueError('Invalid concept path')
    return '/'.join(bytes.fromhex(p[1:]).decode('utf-8') if p.startswith('~') else p for p in parts[1:-1])

def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.025 * 2**attempt)
    finally:
        temporary.unlink(missing_ok=True)

@contextmanager
def update_lock(path):
    """OS lock releases after crash. One update owner across CLI/server processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)

def parse_document(document):
    parts = document.split('---\n', 2)
    if len(parts) != 3 or parts[0]:
        raise ValueError('Missing YAML frontmatter')
    meta = yaml.safe_load(parts[1])
    if not isinstance(meta, dict) or not isinstance(meta.get('type'), str) or not meta['type']:
        raise ValueError('OKF v0.2 requires a type')
    return meta, parts[2]

def render(kind, source, target, status, summary, body):
    meta = {'type': 'Source File' if kind == 'file' else 'Source Directory',
            'title': source or 'Workspace', 'tags': [kind, PurePosixPath(source).suffix.lstrip('.') or 'project'],
            'source_path': source, 'source_fingerprint': target, 'generation_status': status,
            'producer': 'AgentoCode/2', 'okf_version': '0.2'}
    return '---\n' + yaml.safe_dump(meta, allow_unicode=True, sort_keys=True) + '---\n' + \
           '# Explanation\n\n' + summary + '\n\n' + body

def semantic_text(document):
    meta, body = parse_document(document)
    return '\n'.join([str(meta['type']), str(meta.get('title', '')), ' '.join(meta.get('tags', [])),
                      str(meta.get('source_path', '')), body])

class Superseded(RuntimeError):
    pass

class KnowledgeBase:
    def __init__(self, directory, generator=None, index=None, progress=None):
        self.root = Path(directory).resolve(strict=True)
        self.bundle = safe_path(self.root, 'knowledge_base')
        self.state = safe_path(self.root, '.guardian')
        self.generator = generator or GroqGenerator()
        self.index = index or ChromaIndex(self.state / 'vectors')
        self.progress = progress or (lambda message: None)
        self.cancel = threading.Event()
        if isinstance(self.generator, GroqGenerator):
            self.generator.cache_dir = self.state / 'generation-cache'
            self.generator.cancel = self.cancel
        self.workspace_id = digest(str(self.root))

    def _owned(self, create=False):
        marker = self.bundle / '.owner.json'
        expected = {'layout': LAYOUT, 'workspace': self.workspace_id, 'okf': '0.2'}
        if self.bundle.exists():
            if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != expected:
                raise ValueError('Unowned or incompatible knowledge_base; refusing to overwrite')
        elif create:
            self.bundle.mkdir()
            atomic_write(marker, json.dumps(expected))
        else:
            return False
        return True

    def load(self):
        if not self._owned():
            return {'records': {}, 'revision': 0}
        path = self.bundle / '.manifest.json'
        if not path.exists():
            return {'records': {}, 'revision': 0}
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value.get('records'), dict):
            raise ValueError('Invalid manifest; preserve bundle for recovery')
        return value

    def save(self, manifest):
        atomic_write(self.bundle / '.manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))

    def record_id(self, kind, source):
        return digest(self.workspace_id + ':' + kind + ':' + source)

    def snapshot(self):
        files, dirs, skipped = inventory(self.root)
        facts = {p: extract(p, v['source']) for p,v in files.items()}
        dependencies = resolve(facts)
        targets = {}
        for path, value in files.items():
            seen, pending = set(), [path]
            while pending:
                item = pending.pop()
                if item in seen:
                    continue
                seen.add(item)
                pending.extend(dependencies.get(item, []))
            target = digest(json.dumps({p: files[p]['hash'] for p in sorted(seen)}, sort_keys=True))
            targets[('file', path)] = digest(target + self.generator.fingerprint)
        for directory in sorted(dirs, key=lambda p: (-len(PurePosixPath(p).parts), p)):
            # Use explicit parent normalization; filenames containing dots are preserved.
            children = [(k,p,h) for (k,p),h in targets.items()
                        if ('' if PurePosixPath(p).parent == PurePosixPath('.') else str(PurePosixPath(p).parent)) == directory]
            targets[('directory', directory)] = digest(json.dumps(sorted(children)) + self.generator.fingerprint)
        return files, dirs, skipped, facts, dependencies, targets

    def _check_snapshot(self, targets):
        if self.cancel.is_set():
            raise Superseded('Cancelled; pending work resumes on next sync')
        if self.snapshot()[-1] != targets:
            raise Superseded('Source changed during generation; latest revision queued')

    def sync(self):
        with update_lock(self.state / 'update.lock'):
            return self._sync()

    def _sync(self):
        self._owned(create=True)
        manifest = self.load()
        records = manifest['records']
        files, dirs, skipped, facts, dependencies, targets = self.snapshot()
        manifest['skipped'] = skipped
        manifest['revision'] += 1
        current_ids = {self.record_id(k,p) for k,p in targets}
        obsolete = set(records) - current_ids
        for rid in obsolete:
            records[rid]['status'] = 'deleted'
            records[rid]['index_status'] = 'stale'
            safe_path(self.bundle,records[rid]['concept']).unlink(missing_ok=True)
        manifest['pending_deletions'] = sorted(set(manifest.get('pending_deletions',[])) | obsolete)
        for rid in obsolete:
            del records[rid]
        # Invalidate every affected record BEFORE any remote generation or embedding request.
        for (kind, path), target in targets.items():
            rid = self.record_id(kind, path)
            record = records.setdefault(rid, {'kind': kind, 'source': path, 'concept': concept_path(kind,path)})
            if record.get('target') != target:
                record.update(target=target, status='pending', index_status='stale', revision=manifest['revision'])
        self.save(manifest)
        try:
            fingerprint = self.index.fingerprint
            existing_ids = self.index.ids()
            self.index.delete(list(existing_ids - current_ids))
            self.index.delete(manifest['pending_deletions'])
            manifest['pending_deletions'] = []
            manifest.pop('index_error', None)
        except Exception as exc:
            fingerprint, existing_ids = None, set()
            manifest['index_error'] = f'{type(exc).__name__}: {exc}'
        self.save(manifest)
        order = sorted(targets, key=lambda item: (item[0] == 'directory',
            -len(PurePosixPath(item[1]).parts) if item[0] == 'directory' else 0, item[1]))
        for kind, path in order:
            rid = self.record_id(kind,path)
            record = records[rid]
            docpath = safe_path(self.bundle, record['concept'])
            try:
                self._check_snapshot(targets)
                children = [r for r in records.values() if r.get('status') != 'deleted' and
                    r['source'] != path and ('' if PurePosixPath(r['source']).parent == PurePosixPath('.')
                    else str(PurePosixPath(r['source']).parent)) == path] if kind == 'directory' else []
                child_state = digest(json.dumps([(r['concept'],r.get('status'),r.get('doc_hash')) for r in children], sort_keys=True))
                if kind == 'directory' and child_state != record.get('child_state'):
                    record.update(status='pending', index_status='stale')
                    self.save(manifest)
                if record.get('status') != 'complete' or not docpath.exists():
                    self.progress(f'Generating {kind}: {path or "/"}')
                    if not docpath.exists():
                        atomic_write(docpath, render(kind,path,record['target'],'pending','Pending generation.',''))
                    if kind == 'file':
                        material = files[path]['source'] + '\n\nExtracted facts:\n' + json.dumps(facts[path], ensure_ascii=False)
                        related = {p: [{'name': d['name'], 'signature': d['signature'], 'docstring': d['docstring']}
                                      for d in facts[p]['definitions']] for p in dependencies[path]}
                        material += '\nRelated inventory (not executed):\n' + json.dumps(related, ensure_ascii=False)
                        body = '## Extracted facts (not generated prose)\n\n```json\n' + json.dumps(facts[path], ensure_ascii=False, indent=2) + '\n```\n'
                        parent = '' if PurePosixPath(path).parent == PurePosixPath('.') else str(PurePosixPath(path).parent)
                        links = [('directory', parent), *[('file', p) for p in dependencies[path]]]
                    else:
                        material = '\n'.join(f"{r['kind']} {r['source']}: {r.get('summary','Unavailable')[:1000]} [{r.get('status')}]" for r in children)
                        body = '## Immediate children\n\n' + '\n'.join(f"- {r['source']}: {r.get('status')}" for r in children) + '\n'
                        links = [(r['kind'],r['source']) for r in children]
                    summary = self.generator.explain(kind,path,material)
                    if not isinstance(summary, str) or not summary.strip():
                        raise ValueError('Empty generated explanation')
                    body += '\n## Related concepts\n\n' + '\n'.join(
                        f"- [{p or 'Workspace'}]({quote(os.path.relpath(self.bundle/concept_path(k,p), docpath.parent).replace(os.sep,'/'), safe='/.' )})"
                        for k,p in links)
                    document = render(kind,path,record['target'],'complete',summary,body)
                    parse_document(document)
                    self._check_snapshot(targets)
                    atomic_write(docpath, document)
                    record.update(status='complete', index_status='pending', summary=summary[:2000],
                                  doc_hash=digest(document), child_state=child_state, dependencies=dependencies.get(path, []))
                    record.pop('error', None)
                    self.save(manifest)
                document = docpath.read_text(encoding='utf-8')
                meta, _ = parse_document(document)
                if meta.get('source_path') != path or meta.get('source_fingerprint') != record['target'] or meta.get('generation_status') != 'complete':
                    raise ValueError('Knowledge identity/status differs from manifest')
                document_hash = digest(document)
                # Manual body edits are reindexed, never interpreted as source or regeneration events.
                if record.get('doc_hash') != document_hash:
                    record.update(doc_hash=document_hash, index_status='pending',
                                  summary=parse_document(document)[1].split('## Extracted facts')[0][:2000])
                if fingerprint is None:
                    record.update(index_status='failed', index_error=manifest['index_error'])
                elif (record.get('index_status') != 'indexed' or record.get('embedding_fingerprint') != fingerprint
                      or rid not in existing_ids):
                    record['index_status'] = 'pending'
                    self.save(manifest)
                    self._check_snapshot(targets)
                    semantic = semantic_text(document)
                    self.index.upsert(rid, document, semantic, {'source_path': path, 'kind': kind,
                        'concept_path': record['concept'], 'workspace': self.workspace_id, 'target': record['target'],
                        'document_hash': document_hash, 'semantic_hash': digest(semantic), 'embedding_fingerprint': fingerprint})
                    self._check_snapshot(targets)
                    record.update(index_status='indexed', embedding_fingerprint=fingerprint)
                    record.pop('index_error', None)
            except Superseded as exc:
                record.update(status='pending', index_status='stale', error=str(exc))
                self.save(manifest)
                return manifest
            except Exception as exc:
                if record.get('status') == 'complete':
                    record.update(index_status='failed', index_error=f'{type(exc).__name__}: {exc}')
                else:
                    record.update(status='failed', index_status='stale', error=f'{type(exc).__name__}: {exc}')
                self.progress(f'Pending/failed {path or "/"}: {type(exc).__name__}')
            self.save(manifest)
        # Deterministic navigation, including failed concepts. Never vector-index these files.
        listing = '# Knowledge index\n\n' + '\n'.join(f"- [{r['source'] or 'Workspace'}]({r['concept']}) — {r['status']} / {r['index_status']}"
                   for r in records.values() if r['status'] != 'deleted') + '\n'
        atomic_write(self.bundle / 'index.md', listing)
        for record in records.values():
            if record['kind'] != 'directory' or record['status'] == 'deleted':
                continue
            directory = safe_path(self.bundle,record['concept']).parent
            children = [r for r in records.values() if r['source'] != record['source'] and r['status'] != 'deleted'
                        and ('' if PurePosixPath(r['source']).parent == PurePosixPath('.') else str(PurePosixPath(r['source']).parent)) == record['source']]
            listing = '# Immediate concepts\n\n' + '\n'.join(
                f"- [{r['source']}]({quote(os.path.relpath(self.bundle/r['concept'],directory).replace(os.sep,'/'),safe='/.' )}) — {r['status']}"
                for r in children) + '\n'
            atomic_write(directory/'index.md',listing)
        self.save(manifest)
        return manifest

    def status(self):
        manifest = self.load()
        records = list(manifest['records'].values())
        embedder = getattr(self.index,'embedder',None)
        runtime = embedder.status() if embedder is not None and hasattr(embedder,'status') else {}
        return {'concepts': len(records), 'complete': sum(r.get('status') == 'complete' for r in records),
                'indexed': sum(r.get('index_status') == 'indexed' for r in records),
                'failures': {r['source']: r.get('error') or r.get('index_error') for r in records if r.get('error') or r.get('index_error')},
                'index_error': manifest.get('index_error'), 'excluded': manifest.get('skipped', {}),
                'embedding':runtime,'pending_deletions':len(manifest.get('pending_deletions',[]))}
