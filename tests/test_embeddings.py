import math
from dataclasses import replace
from types import SimpleNamespace
import pytest
from pathlib import Path
from embeddings import Profile, SentenceTransformersEmbedder, normalized, ChromaIndex, configured_profile
from generation import GroqGenerator
from workspace import digest

class StubEncoder(SentenceTransformersEmbedder):
    def __init__(self, profile=None):
        self.requests=[]
        owner=self
        class Model:
            max_seq_length=8192
            def __getitem__(self,index):
                if index==0: return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192)))
                return SimpleNamespace(pooling_mode_mean_tokens=True,pooling_mode_lasttoken=True)
            def get_sentence_embedding_dimension(self): return owner.profile.dimensions
            def eval(self): return self
            def parameters(self): return iter([SimpleNamespace(dtype='float32')])
            def tokenizer(self,value,**kwargs):
                owner.requests.append(('tokenize',(value,kwargs)))
                return {'input_ids':list(range(len(value.split())+2))}
            def encode(self,values,**kwargs):
                owner.requests.append(('encode',(values,kwargs)))
                return [[1.0]*owner.profile.dimensions for v in values]
        super().__init__(profile=profile,model_factory=lambda *a,**kw:Model())
        self.asset_root=Path('.guardian/no-test-model-assets')

def test_nomic_asymmetric_templates_and_no_truncation():
    e = StubEncoder()
    assert len(e.embed_documents(['project tail_detail'])[0]) == 768
    e.embed_queries(['project'])
    calls = [data for path,data in e.requests if path=='encode']
    assert calls[0][0][0] == 'search_document: project tail_detail'
    assert calls[1][0][0] == 'search_query: project'
    assert all(data[1]['show_progress_bar'] is False and data[1]['prompt']=='' for data in calls)
    assert all(data[1]['truncation'] is False for name,data in e.requests if name=='tokenize')
    assert e.load_count==1
    assert abs(sum(v*v for v in e.embed_documents(['x'])[0])-1) < 1e-8

def test_qwen_query_instruction_and_document_plain(monkeypatch):
    monkeypatch.setenv('GUARDIAN_EMBED_PROFILE','qwen4b_st'
    )
    monkeypatch.setenv('GUARDIAN_QWEN_REVISION','a'*40)
    e = StubEncoder(configured_profile())
    e.embed_documents(['plain document'])
    e.embed_queries(['query'])
    calls = [data for path,data in e.requests if path=='encode']
    assert calls[0][0][0]=='plain document'
    assert calls[1][0][0].startswith('Instruct: ')
    assert '\nQuery: query' in calls[1][0][0]

@pytest.mark.parametrize('vector,width', [([0.,0.],2),([float('nan'),1],2),([1.],2),([float('inf')],1)])
def test_invalid_vectors(vector,width):
    with pytest.raises(ValueError): normalized(vector,width)

def test_overflow_rejected_and_profile_identity():
    e = StubEncoder(replace(Profile(),context=8))
    with pytest.raises(ValueError,match='tokens; limit'):
        e.embed_documents(['word '*9])
    assert e.fingerprint != StubEncoder().fingerprint
    assert not any(name=='encode' for name,_ in e.requests)

def test_unknown_profile_and_unpinned_qwen_rejected(monkeypatch):
    monkeypatch.setenv('GUARDIAN_EMBED_PROFILE','qwen4b_st')
    monkeypatch.delenv('GUARDIAN_QWEN_REVISION',raising=False)
    with pytest.raises(ValueError,match='pinned'):configured_profile()
    monkeypatch.setenv('GUARDIAN_EMBED_PROFILE','old_service_profile')
    with pytest.raises(ValueError,match='Unsupported'):configured_profile()

def test_real_chroma_storage_with_deterministic_vectors(tmp_path):
    e = StubEncoder()
    index = ChromaIndex(tmp_path/'db',e)
    index.upsert('stable','---\ntype: Source File\n---\nfull text','semantic one',{'source_path':'a.py'})
    index.upsert('stable','updated full text','semantic two',{'source_path':'a.py'})
    assert index.ids() == {'stable'}
    assert index.collection.get(ids=['stable'])['documents']==['updated full text']
    assert index.query('query')[0][0]=='stable'
    assert index.collection.get(include=['embeddings'])['embeddings'].shape == (1,768)
    index.delete(['stable'])
    assert not index.ids()

def test_generation_isolated_and_bounded():
    calls = []
    class Completions:
        def create(self,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"Bounded explanation"}'))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    gen = GroqGenerator(client=client,max_input_bytes=600,retries=0)
    gen.explain('file','a.py','x = "é"\n'*300)
    gen.explain('file','b.py','different')
    assert len(calls)>2
    assert all(len(c['messages'])==2 and 'tools' not in c for c in calls)
    assert all(len(c['messages'][1]['content'].encode())<=600 for c in calls)
    assert 'a.py' not in calls[-1]['messages'][1]['content']

def test_generation_rejects_malformed_output():
    class Completions:
        def create(self,**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"unexpected":1}'))])
    gen = GroqGenerator(client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),retries=0)
    with pytest.raises(ValueError,match='schema'):
        gen.explain('file','a.py','x=1')

def test_generation_cache_resume(tmp_path):
    calls=[]
    class Completions:
        def create(self,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"cached unit"}'))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    gen=GroqGenerator(client=client,retries=0)
    gen.cache_dir=tmp_path/'cache'
    assert gen.explain('file','a.py','x=1')=='cached unit'
    assert gen.explain('file','a.py','x=1')=='cached unit'
    assert len(calls)==1

def test_confirmed_corrupt_index_preserved_and_rebuilt(tmp_path,monkeypatch):
    import chromadb
    import sqlite3
    original=chromadb.PersistentClient
    calls=[]
    def create(**kwargs):
        calls.append(kwargs['path'])
        if len(calls)==1:
            raise sqlite3.DatabaseError('database disk image is malformed')
        return original(**kwargs)
    monkeypatch.setattr(chromadb,'PersistentClient',create)
    base=tmp_path/'vectors'
    base.mkdir()
    (base/'preserved.txt').write_text('original')
    index=ChromaIndex(base,StubEncoder())
    index.upsert('id','full document','semantic text',{'kind':'file'})
    assert index.ids()=={'id'}
    assert (base/'preserved.txt').read_text()=='original'
    reopened=ChromaIndex(base,StubEncoder())
    assert reopened.directory==index.directory
    assert reopened.ids()=={'id'}

def test_generation_preserves_fitting_logical_units():
    inputs=[]
    class Completions:
        def create(self,**kwargs):
            inputs.append(kwargs['messages'][1]['content'])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"summary":"unit"}'))])
    gen=GroqGenerator(client=SimpleNamespace(chat=SimpleNamespace(completions=Completions())),max_input_bytes=600,retries=0)
    source='\n'.join(f'def feature_{i}():\n    value = "'+('x'*140)+f'"\n    return "end_{i}"\n' for i in range(8))
    gen.explain('file','many.py',source)
    assert len(inputs)>1
    for i in range(8):
        part=next(text for text in inputs if f'def feature_{i}(' in text)
        assert f'end_{i}' in part
