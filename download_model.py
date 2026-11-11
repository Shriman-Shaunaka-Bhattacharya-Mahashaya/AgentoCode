"""Fetch and verify the pinned Nomic snapshot without executing model code."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import shutil
from embeddings import Profile
from knowledge import update_lock

def _sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def _curl(args):
    result=subprocess.run(['curl.exe','-L','-f','-sS','--connect-timeout','15',*args],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout

def download():
    profile=Profile()
    root=Path(__file__).resolve().parent/'.guardian/models'/profile.name/profile.revision
    root.mkdir(parents=True,exist_ok=True)
    with update_lock(root/'download.lock'):
        _download_snapshot(profile,root)

def _download_snapshot(profile,root):
    base=f'https://huggingface.co/{profile.model}/resolve/{profile.revision}/'
    files=['config.json','modules.json','sentence_bert_config.json','config_sentence_transformers.json',
        '1_Pooling/config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt']
    for name in files:
        target=root/name
        target.parent.mkdir(exist_ok=True,parents=True)
        temporary=target.with_suffix(target.suffix+'.download')
        _curl(['--max-time','90','-o',str(temporary),base+name])
        os.replace(temporary,target)
    headers=_curl(['--head','--max-time','60',base+'model.safetensors'])
    lengths=re.findall(r'(?im)^(?:content-length|x-linked-size):\s*(\d+)',headers)
    if not lengths or max(map(int,lengths))<=0:
        raise RuntimeError('Missing upstream model size metadata')
    size=max(map(int,lengths))
    hashes=re.findall(r'(?im)^x-linked-etag:\s*"?([a-f0-9]{64})',headers)
    if not hashes:
        raise RuntimeError('Missing upstream model checksum')
    target=root/'model.safetensors'
    cached_ok=target.is_file() and target.stat().st_size==size and _sha(target)==hashes[0]
    if not cached_ok:
        width=(size+7)//8
        count=(size+width-1)//width
        def part(i):
            start=i*width;end=min(size-1,start+width-1)
            path=root/f'weights-{i}.part'
            # A damaged full model invalidates any older partial-cache assumption as well.
            if target.exists() or not path.exists() or path.stat().st_size!=end-start+1:
                _curl(['--max-time','600','--range',f'{start}-{end}','-o',str(path),base+'model.safetensors'])
            if path.stat().st_size!=end-start+1:
                raise RuntimeError('Model range size mismatch')
            print(f'Model download part {i+1}/{count} complete',flush=True)
            return path
        with ThreadPoolExecutor(max_workers=8) as pool:
            parts=list(pool.map(part,range(count)))
        temporary=target.with_suffix('.download')
        try:
            with temporary.open('wb') as output:
                for path in parts:
                    with path.open('rb') as source:
                        shutil.copyfileobj(source,output,1024*1024)
            if _sha(temporary)!=hashes[0]:
                for path in parts:
                    assert path.resolve().parent==root.resolve()
                    path.unlink(missing_ok=True)
                raise RuntimeError('Model checksum mismatch; invalid partials removed, retry download')
            os.replace(temporary,target)
        finally:
            temporary.unlink(missing_ok=True)
    identity={'model':profile.model,'revision':profile.revision,
              'files':{name:_sha(root/name) for name in [*files,'model.safetensors']}}
    temporary=root/'asset-manifest.json.download'
    temporary.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    os.replace(temporary,root/'asset-manifest.json')
    print('Verified pinned Nomic snapshot: '+str(root),flush=True)

if __name__=='__main__':
    download()
