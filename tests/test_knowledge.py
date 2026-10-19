import json
from pathlib import Path
import pytest
from conftest import FakeGenerator, FakeIndex
from facts import extract, resolve
from knowledge import KnowledgeBase, concept_path, source_path, parse_document, semantic_text, update_lock
from rag import RAGManager
from workspace import inventory, safe_path, digest
from server import WorkspaceTools

@pytest.mark.parametrize('source',['a.py','a.js','content.md','index.md','log.md','%/content.md','nested/a b.py','nested/é.py','~escape/FILE.PY'])
def test_mapping_round_trip(source):
    for kind in ('file','directory'):
        assert source_path(concept_path(kind,source)) == source
    assert concept_path('file',source) != concept_path('directory',source)

def test_unowned_bundle_preserved(project):
    p = project/'knowledge_base'
    p.mkdir()
    (p/'keep.txt').write_text('keep')
    with pytest.raises(ValueError,match='Unowned'):
        KnowledgeBase(project).sync()
    assert (p/'keep.txt').read_text() == 'keep'

def test_unicode_facts_nested_and_parse_errors():
    facts = extract('x.py','"é"\nclass C:\n    def café(self, x: str = "é") -> str:\n        return x\n')
    assert facts['definitions'][1]['name'] == 'C.café'
    assert facts['definitions'][1]['line'] == 3
    assert 'x: str' in facts['definitions'][1]['signature']
    assert 'parse error' in extract('x.py','def (')['limitations'][0]
    assert 'Text description' in extract('x.js','function x() {}')['limitations'][0]

def test_imports_cycles_and_conservative_calls():
    facts = {'a.py':extract('a.py','from b import f\ndef g():\n    f()\n'),
             'b.py':extract('b.py','from a import g\ndef f():\n    g()\n')}
    deps = resolve(facts)
    assert deps == {'a.py':['b.py'],'b.py':['a.py']}
    assert facts['a.py']['calls'][0]['target'] == 'unresolved/external/dynamic'
    assert facts['a.py']['calls'][0]['candidate'].startswith('b.py:')

def test_first_build_and_unchanged_restart(kb):
    manifest = kb.sync()
    assert len(manifest['records']) == 4
    assert all(r['status']=='complete' and r['index_status']=='indexed' for r in manifest['records'].values())
    assert kb.generator.calls[-1] == ('directory','')
    for record in manifest['records'].values():
        meta,body = parse_document((kb.bundle/record['concept']).read_text(encoding='utf-8'))
        assert meta['type'] and body
    counts = len(kb.generator.calls),len(kb.index.calls)
    kb.sync()
    restarted = RAGManager(kb.root,generator=kb.generator,index=kb.index)
    restarted.sync()
    assert counts == (len(kb.generator.calls),len(kb.index.calls))

@pytest.mark.parametrize('actor',['user','agent'])
def test_update_stable_ids_and_dependency_refresh(kb,actor):
    kb.sync()
    ids = kb.index.ids()
    before = list(kb.index.calls)
    source = (kb.root/'calculator.py').read_text()+'\n# updated source\n'
    if actor == 'agent':
        WorkspaceTools(kb.root,kb).write_file('calculator.py',source)
    else:
        (kb.root/'calculator.py').write_text(source)
    # Source validation blocks old knowledge even before the worker processes the event.
    result = kb.search('calculator',token_budget=12000)
    assert 'KNOWLEDGE calculator.py' not in result
    kb.sync()
    assert kb.index.ids() == ids
    changed = set(kb.index.calls[len(before):])
    assert kb.record_id('file','calculator.py') in changed
    assert kb.record_id('file','main.py') in changed
    assert kb.record_id('directory','') in changed
    assert kb.record_id('file','README.md') not in changed
    # Fake explanations deliberately stay identical; vectors are still recomputed.
    counts = len(kb.generator.calls),len(kb.index.calls)
    kb.sync()
    assert counts == (len(kb.generator.calls),len(kb.index.calls))

def test_add_move_delete_empty_dirs(kb):
    kb.sync()
    (kb.root/'nested').mkdir()
    (kb.root/'nested'/'new.py').write_text('def new(): return 1\n')
    kb.sync()
    assert len(kb.index.ids()) == 6
    old = kb.record_id('file','nested/new.py')
    (kb.root/'nested'/'new.py').rename(kb.root/'nested'/'moved.py')
    kb.sync()
    assert old not in kb.index.ids()
    (kb.root/'nested'/'moved.py').unlink()
    kb.sync()
    assert len(kb.index.ids()) == 5
    assert kb.record_id('directory','nested') in kb.index.ids()

def test_failed_generation_partial_parent_retry(kb):
    kb.generator.fail.add('calculator.py')
    kb.sync()
    assert kb.record_id('file','calculator.py') not in kb.index.ids()
    assert kb.load()['records'][kb.record_id('directory','')]['status'] == 'complete'
    kb.generator.fail.clear()
    kb.sync()
    assert len(kb.index.ids()) == 4
    assert all(r['index_status']=='indexed' for r in kb.load()['records'].values())

def test_upsert_failure_resume_without_regeneration(kb):
    kb.index.fail = True
    kb.sync()
    calls = len(kb.generator.calls)
    assert kb.status()['indexed'] == 0
    kb.index.fail = False
    kb.sync()
    assert kb.status()['indexed'] == 4
    assert len(kb.generator.calls) == calls

@pytest.mark.parametrize('stage',['generation','index'])
def test_superseded_revision_cannot_be_current(kb,stage):
    def edit():
        (kb.root/'calculator.py').write_text('def latest(): return 99\n')
    if stage == 'generation':
        kb.generator.hook = edit
    else:
        kb.index.hook = edit
    kb.sync()
    assert kb.status()['indexed'] < 4
    kb.sync()
    rec = kb.load()['records'][kb.record_id('file','calculator.py')]
    assert rec['target'] == kb.snapshot()[-1][('file','calculator.py')]
    assert 'latest' in (kb.bundle/rec['concept']).read_text(encoding='utf-8')

def test_manual_edit_reindex_and_missing_index_recovery(kb):
    kb.sync()
    calls = len(kb.generator.calls)
    record = kb.load()['records'][kb.record_id('file','calculator.py')]
    path = kb.bundle/record['concept']
    path.write_text(path.read_text(encoding='utf-8')+'\nManual explanation.\n',encoding='utf-8')
    kb.sync()
    assert len(kb.generator.calls) == calls + 1  # parent refreshes its changed child explanation
    assert 'Manual explanation' in kb.index.records[kb.record_id('file','calculator.py')][0]
    calls = len(kb.generator.calls)
    kb.index.records.clear()
    kb.sync()
    assert len(kb.index.records) == 4
    assert len(kb.generator.calls) == calls

def test_profile_change_reindexes_without_generation(kb):
    kb.sync()
    calls = len(kb.generator.calls)
    kb.index.fingerprint = 'different-profile'
    kb.sync()
    assert len(kb.generator.calls) == calls
    assert all(v[2]['embedding_fingerprint']=='different-profile' for v in kb.index.records.values())

def test_context_budget_source_mapping_no_duplicates(kb):
    kb.sync()
    result = kb.search('divide calculator',token_budget=1800)
    assert len(result.encode('utf-8')) <= 1800
    assert 'CURRENT SOURCE calculator.py' in result
    assert result.count('CURRENT SOURCE calculator.py') == 1
    assert 'Cannot divide by zero' in kb.search('zero',token_budget=12000)

def test_semantic_serializer_omits_operational_fields(kb):
    kb.sync()
    for rid,(document,semantic,metadata) in kb.index.records.items():
        assert metadata['target'] not in semantic
        assert metadata['target'] in document
        assert 'source_fingerprint:' not in semantic

def test_exclusions_and_binary(project):
    (project/'.env').write_text('secret')
    (project/'venv').mkdir()
    (project/'venv'/'secret.py').write_text('secret')
    (project/'image.bin').write_bytes(b'\0\x00')
    files,_,skipped = inventory(project)
    assert '.env' not in files and 'venv/secret.py' not in files
    assert skipped['image.bin'] == 'Binary input'

def test_single_update_owner(kb):
    with update_lock(kb.state/'update.lock'):
        with pytest.raises(OSError):
            kb.sync()

def test_missing_index_visible_fallback(kb):
    kb.sync()
    def fail(*a,**kw):
        raise RuntimeError('offline')
    kb.index.query = fail
    result = kb.search('factorial',token_budget=12000)
    assert 'Semantic index unavailable' in result
    assert 'CURRENT SOURCE calculator.py' in result

def test_corrupt_manifest_keeps_source_fallback(kb):
    kb.sync()
    (kb.bundle/'.manifest.json').write_text('{broken',encoding='utf-8')
    assert 'CURRENT SOURCE calculator.py' in kb.search('factorial')
    with pytest.raises(ValueError):
        kb.sync()
    assert (kb.bundle/'.manifest.json').read_text()=='{broken'

def test_interrupted_document_publication_resumes(kb,monkeypatch):
    import knowledge
    original = knowledge.atomic_write
    def broken(path,text):
        if str(path).endswith('content.md') and 'generation_status: complete' in text:
            raise OSError('Injected publication failure')
        return original(path,text)
    monkeypatch.setattr(knowledge,'atomic_write',broken)
    kb.sync()
    assert kb.status()['indexed']==0
    monkeypatch.setattr(knowledge,'atomic_write',original)
    kb.sync()
    assert kb.status()['indexed']==4

def test_orphan_records_deleted(kb):
    kb.sync()
    kb.index.records['orphan']=('unused','unused',{})
    kb.sync()
    assert 'orphan' not in kb.index.records

def test_nested_ignores_and_size_limit(project):
    (project/'.gitignore').write_text('ignored.txt\n')
    (project/'ignored.txt').write_text('private')
    (project/'large.txt').write_text('x'*513000)
    (project/'sub').mkdir()
    (project/'sub'/'.gitignore').write_text('*.txt\n!keep.txt\n')
    (project/'sub'/'skip.txt').write_text('skip')
    (project/'sub'/'keep.txt').write_text('keep')
    files,_,skipped=inventory(project)
    assert 'ignored.txt' not in files and 'sub/skip.txt' not in files
    assert 'sub/keep.txt' in files
    assert 'large.txt' in skipped

def test_ignore_fallback_without_git(project,monkeypatch):
    import workspace
    (project/'.gitignore').write_text('secret.txt\n')
    (project/'secret.txt').write_text('private')
    def unavailable(*args,**kwargs): raise OSError('git absent')
    monkeypatch.setattr(workspace.subprocess,'run',unavailable)
    assert 'secret.txt' not in inventory(project)[0]

def test_larger_nested_fixture_tail_and_unrelated_queries(kb):
    (kb.root/'package').mkdir()
    (kb.root/'package'/'deep').mkdir()
    for n in range(12):
        (kb.root/'package'/f'module{n}.py').write_text(f'def feature{n}():\n    return {n}\n')
    (kb.root/'package'/'deep'/'large.py').write_text(
        '\n'.join(f'def function_{i}(): return {i}\n' for i in range(300))+
        '\nTAIL_UNIQUE_MARKER = "final detail"\n',encoding='utf-8')
    (kb.root/'package'/'deep'/'broken.py').write_text('def broken(')
    (kb.root/'package'/'deep'/'content.md').write_text('说明：计算历史',encoding='utf-8')
    kb.sync()
    assert kb.status()['concepts']==21
    assert kb.status()['indexed']==21
    result=kb.search('TAIL_UNIQUE_MARKER',token_budget=1600)
    assert 'TAIL_UNIQUE_MARKER' in result and len(result.encode('utf-8'))<=1600
    kb.index.query=lambda *args,**kwargs: []
    assert 'No current relevant evidence' in kb.search('unrelated_xyz_giraffe')
    assert '说明' in kb.search('说明',token_budget=1600)

def test_failed_replacement_vector_is_not_a_semantic_candidate(kb):
    kb.sync()
    rid=kb.record_id('file','calculator.py')
    (kb.root/'calculator.py').write_text('def changed(): return 1\n')
    kb.index.fail=True
    kb.sync()
    assert rid in kb.index.records  # old vector intentionally remains persisted
    assert kb.load()['records'][rid]['index_status']=='failed'
    kb.index.query=lambda *args,**kwargs: [(rid,0.01)]
    result=kb.search('a paraphrase without literal matches')
    assert 'KNOWLEDGE calculator.py' not in result
    assert 'No current relevant evidence' in result

def test_delete_local_concept_without_index_and_retry_vector_cleanup(kb):
    kb.sync()
    rid=kb.record_id('file','README.md')
    concept=kb.bundle/kb.load()['records'][rid]['concept']
    (kb.root/'README.md').unlink()
    original_ids=kb.index.ids
    def unavailable():raise RuntimeError('index offline')
    kb.index.ids=unavailable
    manifest=kb.sync()
    assert rid not in manifest['records'] and not concept.exists()
    assert rid in manifest['pending_deletions'] and rid in kb.index.records
    assert kb.status()['concepts']==3
    kb.index.ids=original_ids
    manifest=kb.sync()
    assert manifest['pending_deletions']==[] and rid not in kb.index.records

def test_atomic_replacement_retries_transient_permission_error(tmp_path,monkeypatch):
    import knowledge
    path=tmp_path/'state.json';path.write_text('old')
    original=knowledge.os.replace;attempts=[]
    def replacement(source,target):
        attempts.append(target)
        if len(attempts)<3:raise PermissionError('temporary reader')
        original(source,target)
    monkeypatch.setattr(knowledge.os,'replace',replacement)
    monkeypatch.setattr(knowledge.time,'sleep',lambda _:None)
    knowledge.atomic_write(path,'new')
    assert path.read_text()=='new' and len(attempts)==3
