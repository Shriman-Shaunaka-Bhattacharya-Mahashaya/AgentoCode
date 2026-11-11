"""Fresh-process startup and persisted knowledge contracts without live services."""
import os
import subprocess
import sys

import pytest

from conftest import FakeGenerator
from embeddings import ChromaIndex
from rag import RAGManager
from test_embeddings import StubEncoder
from workspace import APP_ROOT


@pytest.mark.parametrize('scenario', ['help', 'missing_key', 'own_workspace'])
def test_cli_fresh_process_without_credentials_or_model(project, scenario):
    env = dict(os.environ, GROQ_API_KEY='', PYTHON_DOTENV_DISABLED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               GUARDIAN_LOCAL_FILES_ONLY='1', PYTHONUTF8='1')
    arguments = {
        'help': ['--help'],
        'missing_key': ['--dir', str(project)],
        'own_workspace': ['--dir', str(APP_ROOT)],
    }[scenario]
    before = {p.relative_to(project): p.read_bytes() for p in project.rglob('*')
              if p.is_file() and '.git' not in p.relative_to(project).parts}
    result = subprocess.run([sys.executable, str(APP_ROOT/'cli.py'), *arguments],
                            cwd=project, env=env, capture_output=True,
                            text=True, encoding='utf-8', timeout=30)
    output = result.stdout + result.stderr
    if scenario == 'help':
        assert result.returncode == 0
        assert '--single-agent' in output and '--workers' in output
    else:
        assert result.returncode != 0
        expected = 'GROQ_API_KEY is missing' if scenario == 'missing_key' else 'agent application'
        assert expected in output
    after = {p.relative_to(project): p.read_bytes() for p in project.rglob('*')
             if p.is_file() and '.git' not in p.relative_to(project).parts}
    assert before == after
    assert not (project/'.guardian').exists()
    assert not (project/'knowledge_base').exists()


def test_real_chroma_lifecycle_survives_reopen_and_recreated_source(project):
    directory = project/'.guardian/vectors'
    encoder = StubEncoder()
    knowledge = RAGManager(project, generator=FakeGenerator(),
                           index=ChromaIndex(directory, encoder))
    knowledge.sync()
    initial_ids = knowledge.index.ids()
    rid = knowledge.record_id('file', 'calculator.py')
    sibling = knowledge.record_id('file', 'README.md')
    initial = knowledge.index.collection.get(ids=[rid, sibling], include=['metadatas'])
    metadata = dict(zip(initial['ids'], initial['metadatas']))
    # No-op synchronization must neither regenerate nor call the encoder.
    before = (len(knowledge.generator.calls), encoder.document_calls)
    knowledge.sync()
    assert before == (len(knowledge.generator.calls), encoder.document_calls)

    path = project/'calculator.py'
    path.write_text('def refreshed_calculation(): return 23\n', encoding='utf-8')
    stale = knowledge.search('refreshed_calculation', structured=True)
    assert '23' in stale['context']
    assert knowledge.status()['stale'] > 0
    knowledge.sync()
    assert knowledge.index.ids() == initial_ids
    changed = knowledge.index.collection.get(ids=[rid, sibling], include=['metadatas'])
    changed_metadata = dict(zip(changed['ids'], changed['metadatas']))
    assert changed_metadata[rid]['target'] != metadata[rid]['target']
    assert changed_metadata[sibling] == metadata[sibling]

    reopened_encoder = StubEncoder()
    reopened = RAGManager(project, generator=FakeGenerator(),
                          index=ChromaIndex(directory, reopened_encoder))
    reopened.sync()
    assert not reopened.generator.calls and reopened_encoder.document_calls == 0
    assert reopened.status()['indexed'] == len(initial_ids)
    path.unlink()
    reopened.sync()
    assert rid not in reopened.index.ids()
    assert rid not in reopened.load()['records']
    assert not (reopened.bundle/'files/calculator.py/content.md').exists()
    path.write_text('def recreated_calculation(): return 31\n', encoding='utf-8')
    reopened.sync()
    assert reopened.index.ids() == initial_ids
    record = reopened.load()['records'][rid]
    stored = reopened.index.collection.get(ids=[rid], include=['documents', 'metadatas'])
    assert 'recreated_calculation' in stored['documents'][0]
    assert stored['metadatas'][0]['document_hash'] == record['doc_hash']
    assert reopened.status()['stale'] == 0
    assert not reopened.load().get('pending_deletions')
