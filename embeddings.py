"""Pinned in-process Sentence Transformers embeddings with strict token preflight."""
from dataclasses import dataclass, asdict
import json
import math
import os
import threading
import importlib.metadata
import uuid
from pathlib import Path
from workspace import digest

@dataclass(frozen=True)
class Profile:
    name: str = 'nomic_v15_st'
    model: str = 'nomic-ai/nomic-embed-text-v1.5'
    revision: str = 'e9b6763023c676ca8431644204f50c2b100d9aab'
    dimensions: int = 768
    context: int = 2048
    document_prefix: str = 'search_document: '
    query_prefix: str = 'search_query: '
    pooling: str = 'mean'
    overflow: str = 'reject'
    serializer: str = 'semantic-v1'
    normalization: str = 'l2'
    metric: str = 'cosine'

def configured_profile():
    choice = os.getenv('GUARDIAN_EMBED_PROFILE', 'nomic_v15_st')
    if choice == 'nomic_v15_st':
        return Profile()
    if choice == 'qwen4b_st':
        revision = os.getenv('GUARDIAN_QWEN_REVISION','')
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Optional Qwen profile requires a pinned 40-character GUARDIAN_QWEN_REVISION')
        return Profile(name=choice, model='Qwen/Qwen3-Embedding-4B', revision=revision, dimensions=2560,
                       document_prefix='', query_prefix='Instruct: Retrieve project knowledge relevant to the question.\nQuery: ',
                       pooling='lasttoken')
    raise ValueError('Unsupported embedding profile; select nomic_v15_st or qwen4b_st')

def normalized(vector, dimensions):
    if len(vector) != dimensions or any(not math.isfinite(v) for v in vector):
        raise ValueError('Invalid embedding width or nonfinite value')
    norm = math.sqrt(sum(v*v for v in vector))
    if not norm:
        raise ValueError('Zero embedding')
    return [float(v/norm) for v in vector]

class SentenceTransformersEmbedder:
    def __init__(self, profile=None, model_factory=None):
        self.profile = profile or configured_profile()
        self.device = os.getenv('GUARDIAN_EMBED_DEVICE','cpu')
        if self.device not in ('cpu','cuda'):
            raise ValueError('GUARDIAN_EMBED_DEVICE must be cpu or cuda')
        self.model_factory = model_factory
        self.model = None
        self.lock = threading.RLock()
        self._fingerprint = None
        self.load_count = 0
        self.document_calls = 0
        self.query_calls = 0
        self.state = 'not_loaded'
        self.last_error = None
        self.asset_root = Path(__file__).resolve().parent / '.guardian' / 'models'

    def _load(self):
        if self.model is not None:
            return
        self.state = 'loading'
        try:
            factory = self.model_factory
            if factory is None:
                os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
                os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
                from sentence_transformers import SentenceTransformer
                import torch
                torch.set_num_threads(4)
                factory = SentenceTransformer
            local = self.asset_root / self.profile.name / self.profile.revision
            kwargs = dict(device=self.device, trust_remote_code=False,
                cache_folder=str(self.asset_root/'hub'), backend='torch',
                model_kwargs={'dtype':'float32' if self.device == 'cpu' else 'float16'})
            if (local/'asset-manifest.json').is_file():
                identity = json.loads((local/'asset-manifest.json').read_text(encoding='utf-8'))
                if identity.get('revision') != self.profile.revision or identity.get('model') != self.profile.model:
                    raise ValueError('Model asset identity mismatch')
                # Verify local files before trusting a pinned local snapshot.
                import hashlib
                for name, expected in identity['files'].items():
                    path = (local/name).resolve()
                    if not path.is_relative_to(local.resolve()):
                        raise ValueError('Invalid asset path')
                    with path.open('rb') as handle:
                        if hashlib.file_digest(handle,'sha256').hexdigest() != expected:
                            raise ValueError('Model asset checksum mismatch')
                model = factory(str(local), local_files_only=True, **kwargs)
                self.asset_identity = identity
            else:
                model = factory(self.profile.model, revision=self.profile.revision,
                    local_files_only=os.getenv('GUARDIAN_LOCAL_FILES_ONLY','0') == '1', **kwargs)
                self.asset_identity = {'model':self.profile.model,'revision':self.profile.revision}
            cap = int(model.max_seq_length)
            positional = int(model[0].auto_model.config.max_position_embeddings)
            tokenizer_cap = getattr(model.tokenizer,'model_max_length',cap)
            tokenizer_cap = tokenizer_cap if isinstance(tokenizer_cap,int) and tokenizer_cap < 1_000_000 else cap
            if self.profile.context > min(cap,positional,tokenizer_cap):
                raise ValueError('Requested context exceeds verified model/tokenizer configuration')
            dimension_method = getattr(model,'get_embedding_dimension',None) or model.get_sentence_embedding_dimension
            if dimension_method() != self.profile.dimensions:
                raise ValueError('Checkpoint embedding width mismatch')
            pooling = model[1]
            if getattr(pooling,'pooling_mode',None) != self.profile.pooling and not getattr(pooling, 'pooling_mode_' + self.profile.pooling + '_tokens', False) and not (
                    self.profile.pooling == 'lasttoken' and getattr(pooling,'pooling_mode_lasttoken',False)):
                raise ValueError('Checkpoint native pooling differs from profile')
            model.max_seq_length = self.profile.context
            model.eval()
            self.model = model
            self.load_count += 1
            self.state = 'ready'
            self.last_error = None
        except Exception as exc:
            self.state = 'failed'
            self.last_error = f'{type(exc).__name__}: {exc}'
            raise

    @property
    def fingerprint(self):
        with self.lock:
            self._load()
            if self._fingerprint is None:
                versions = {name:importlib.metadata.version(name) for name in
                    ('sentence-transformers','transformers','torch','tokenizers')}
                self._fingerprint = digest(json.dumps({'profile':asdict(self.profile),
                    'assets':self.asset_identity,'runtime':versions,'device':self.device,
                    'dtype':str(next(self.model.parameters()).dtype),'backend':'torch'},sort_keys=True))
            return self._fingerprint

    def _embed(self, texts, prefix):
        with self.lock:
            _ = self.fingerprint
            rendered = [prefix + text for text in texts]
            for value in rendered:
                if len(value.encode('utf-8')) > 512_000:
                    raise ValueError('Embedding input exceeds byte limit')
                ids = self.model.tokenizer(value, truncation=False, add_special_tokens=True)['input_ids']
                if len(ids) > self.profile.context:
                    raise ValueError(f'Input has {len(ids)} tokens; limit {self.profile.context}; concept remains unindexed')
            # Explicit empty prompt prevents saved/default prompts from duplicating our prefixes.
            vectors = self.model.encode(rendered, prompt='', batch_size=1, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True)
            if len(vectors) != len(texts):
                raise ValueError('Embedding response count mismatch')
            return [normalized(vector,self.profile.dimensions) for vector in vectors]

    def embed_documents(self, texts):
        vectors = self._embed(texts, self.profile.document_prefix)
        self.document_calls += len(texts)
        return vectors

    def embed_queries(self, texts):
        vectors = self._embed(texts, self.profile.query_prefix)
        self.query_calls += len(texts)
        return vectors

    def status(self):
        return {'runtime':'sentence-transformers','model':self.profile.model,'revision':self.profile.revision,
            'device':self.device,'state':self.state,'load_count':self.load_count,
            'document_calls':self.document_calls,'query_calls':self.query_calls,'error':self.last_error}

class ChromaIndex:
    def __init__(self, directory, embedder=None):
        self.directory = Path(directory)
        self.pointer = self.directory.parent / 'vector-store.json'
        if self.pointer.exists():
            name = json.loads(self.pointer.read_text(encoding='utf-8'))['name']
            if Path(name).name != name or not name.startswith('vectors-recovery-'):
                raise ValueError('Invalid vector store location')
            self.directory = self.directory.parent / name
        self.embedder = embedder or SentenceTransformersEmbedder()
        self._collection = None

    @property
    def fingerprint(self):
        return self.embedder.fingerprint

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings
            def open_store():
                self.client = chromadb.PersistentClient(path=str(self.directory), settings=Settings(anonymized_telemetry=False))
                return self.client.get_or_create_collection('okf_' + self.fingerprint[:40],
                    embedding_function=None, metadata={'hnsw:space': 'cosine', 'embedding_fingerprint': self.fingerprint})
            try:
                self._collection = open_store()
            except Exception as exc:
                if not any(term in str(exc).lower() for term in ('database disk image is malformed', 'not a database')):
                    raise
                # Preserve corrupt files for diagnosis; rebuild from validated documents into a new store.
                self.directory = self.pointer.parent / ('vectors-recovery-' + uuid.uuid4().hex)
                self._collection = open_store()
                temporary = self.pointer.with_suffix('.tmp')
                temporary.write_text(json.dumps({'name':self.directory.name}),encoding='utf-8')
                os.replace(temporary,self.pointer)
        return self._collection

    def upsert(self, record_id, document, semantic, metadata):
        vectors = self.embedder.embed_documents([semantic])
        self.collection.upsert(ids=[record_id], documents=[document], embeddings=vectors, metadatas=[metadata])

    def delete(self, ids):
        if ids:
            self.collection.delete(ids=ids)

    def ids(self):
        return set(self.collection.get(include=[])['ids'])

    def query(self, query, count=12):
        count = min(count, self.collection.count())
        if not count:
            return []
        result = self.collection.query(query_embeddings=self.embedder.embed_queries([query]), n_results=count)
        return list(zip(result['ids'][0], result['distances'][0]))
