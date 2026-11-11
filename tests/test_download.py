"""Exercise bootstrap resume, checksum failure and publication without network I/O."""
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import json
import pytest
import download_model
from embeddings import Profile

@pytest.fixture
def bootstrap(tmp_path,monkeypatch):
    monkeypatch.setattr(download_model,'__file__',str(tmp_path/'download_model.py'))
    profile=Profile();root=tmp_path/'.guardian/models'/profile.name/profile.revision
    root.mkdir(parents=True)
    weights=bytes(range(64));checksum=hashlib.sha256(weights).hexdigest()
    state=NS(root=root,weights=weights,calls=[],headers=f'Content-Length: 64\nx-linked-etag: "{checksum}"\n',fail=False,short=False)
    def curl(command,**kwargs):
        state.calls.append(command)
        if state.fail:return NS(returncode=28,stdout='',stderr='injected network timeout')
        if '--head' in command:return NS(returncode=0,stdout=state.headers,stderr='')
        target=Path(command[command.index('-o')+1])
        if '--range' in command:
            start,end=map(int,command[command.index('--range')+1].split('-'))
            target.write_bytes(weights[start:end if state.short else end+1])
        else:target.write_text('{}')
        return NS(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(download_model.subprocess,'run',curl)
    return state

def test_verified_cache_is_reused_without_weight_download(bootstrap):
    download_model.download()
    model=bootstrap.root/'model.safetensors';before=model.stat().st_mtime_ns
    bootstrap.calls.clear();download_model.download()
    assert model.read_bytes()==bootstrap.weights and model.stat().st_mtime_ns==before
    assert not any('--range' in command for command in bootstrap.calls)
    manifest=json.loads((bootstrap.root/'asset-manifest.json').read_text())
    assert manifest['files']['model.safetensors']==hashlib.sha256(bootstrap.weights).hexdigest()

def test_bad_resumed_partial_is_removed_and_next_attempt_recovers(bootstrap):
    (bootstrap.root/'weights-0.part').write_bytes(b'x'*8)
    with pytest.raises(RuntimeError,match='checksum mismatch'):download_model.download()
    assert not list(bootstrap.root.glob('*.part'))
    assert not (bootstrap.root/'model.safetensors').exists()
    assert not (bootstrap.root/'asset-manifest.json').exists()
    download_model.download()
    assert (bootstrap.root/'model.safetensors').read_bytes()==bootstrap.weights

@pytest.mark.parametrize('headers',['x-linked-etag: "'+'a'*64+'"\n','Content-Length: 64\n'])
def test_missing_upstream_identity_rejected(bootstrap,headers):
    bootstrap.headers=headers
    with pytest.raises(RuntimeError,match='Missing upstream'):download_model.download()
    assert not (bootstrap.root/'asset-manifest.json').exists()

def test_short_range_response_never_publishes_model(bootstrap):
    bootstrap.short=True
    with pytest.raises(RuntimeError,match='range size'):download_model.download()
    assert not (bootstrap.root/'model.safetensors').exists()
    assert not (bootstrap.root/'asset-manifest.json').exists()

def test_network_failure_preserves_existing_model(bootstrap):
    model=bootstrap.root/'model.safetensors';model.write_bytes(bootstrap.weights)
    bootstrap.fail=True
    with pytest.raises(RuntimeError,match='network timeout'):download_model.download()
    assert model.read_bytes()==bootstrap.weights

def test_concurrent_bootstrap_owner_is_rejected_before_download(bootstrap):
    with download_model.update_lock(bootstrap.root/'download.lock'):
        with pytest.raises(OSError):download_model.download()
    assert not bootstrap.calls
