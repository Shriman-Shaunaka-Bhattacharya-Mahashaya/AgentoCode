"""Fetch the selected pinned Nomic snapshot; no model code is executed here."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import re
import subprocess
import shutil
from embeddings import Profile

def download():
    profile = Profile()
    root = Path(__file__).resolve().parent/'.guardian/models'/profile.name/profile.revision
    root.mkdir(parents=True,exist_ok=True)
    base = f'https://huggingface.co/{profile.model}/resolve/{profile.revision}/'
    files = ['config.json','modules.json','sentence_bert_config.json','config_sentence_transformers.json',
        '1_Pooling/config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt']
    def curl(args):
        result=subprocess.run(['curl.exe','-L','-f','-sS','--connect-timeout','15',*args],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        return result.stdout
    for name in files:
        target=root/name
        target.parent.mkdir(exist_ok=True,parents=True)
        curl(['--max-time','90','-o',str(target),base+name])
    headers=curl(['--head','--max-time','60',base+'model.safetensors'])
    lengths=re.findall(r'(?im)^(?:content-length|x-linked-size):\s*(\d+)',headers)
    size=max(map(int,lengths))
    hashes=re.findall(r'(?im)^x-linked-etag:\s*"?([a-f0-9]{64})',headers)
    if not hashes:
        raise RuntimeError('Missing upstream model checksum')
    target=root/'model.safetensors'
    if not target.exists() or target.stat().st_size != size:
        width=(size+7)//8
        def part(i):
            start=i*width; end=min(size-1,start+width-1)
            p=root/f'weights-{i}.part'
            if not p.exists() or p.stat().st_size != end-start+1:
                curl(['--max-time','600','--range',f'{start}-{end}','-o',str(p),base+'model.safetensors'])
            if p.stat().st_size != end-start+1:
                raise RuntimeError('Model range size mismatch')
            print(f'Model download part {i+1}/8 complete',flush=True)
            return p
        with ThreadPoolExecutor(max_workers=8) as pool:
            parts=list(pool.map(part,range(8)))
        with target.open('wb') as output:
            for p in parts:
                with p.open('rb') as source:
                    shutil.copyfileobj(source,output,1024*1024)
    def sha(path):
        with path.open('rb') as handle:
            return hashlib.file_digest(handle,'sha256').hexdigest()
    if sha(target) != hashes[0]:
        raise RuntimeError('Model checksum differs from Hugging Face LFS metadata')
    identity={'model':profile.model,'revision':profile.revision,
        'files':{name:sha(root/name) for name in [*files,'model.safetensors']}}
    (root/'asset-manifest.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    print('Verified pinned Nomic snapshot: '+str(root),flush=True)

if __name__ == '__main__':
    download()
