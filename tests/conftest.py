from pathlib import Path
import shutil
import subprocess
import pytest
from rag import RAGManager

def pytest_configure(config):
    (Path(__file__).resolve().parents[1]/'.guardian').mkdir(exist_ok=True)

class FakeGenerator:
    fingerprint = 'fake-generation-v1'
    def __init__(self):
        self.calls = []
        self.fail = set()
        self.hook = None
    def explain(self, kind, path, material):
        self.calls.append((kind,path))
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        if path in self.fail:
            raise RuntimeError('Injected generation failure')
        return f'Explanation of {kind} {path or "workspace"}. Arithmetic and calculation history.'

class FakeIndex:
    fingerprint = 'fake-embedding-v1'
    def __init__(self):
        self.records = {}
        self.calls = []
        self.fail = False
        self.hook = None
    def ids(self):
        return set(self.records)
    def delete(self, ids):
        for rid in ids:
            self.records.pop(rid,None)
    def upsert(self, rid, document, semantic, metadata):
        self.calls.append(rid)
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        if self.fail:
            raise RuntimeError('Injected index failure')
        self.records[rid] = (document,semantic,metadata)
    def query(self, query, count=12):
        return [(rid,0.1) for rid in list(self.records)[:count]]

@pytest.fixture
def project(tmp_path):
    root = tmp_path/'project'
    root.mkdir()
    subprocess.run(['git','init','-q',str(root)],check=True,capture_output=True)
    source = Path(__file__).resolve().parents[1]/'sample_project'
    for name in ('calculator.py','main.py','README.md'):
        shutil.copy2(source/name,root/name)
    return root

@pytest.fixture
def kb(project):
    return RAGManager(project,generator=FakeGenerator(),index=FakeIndex())
