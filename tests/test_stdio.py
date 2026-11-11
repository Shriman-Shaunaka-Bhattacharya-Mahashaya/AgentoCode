"""Observe raw stdout, including Python/native noise during a failing model load."""
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from workspace import APP_ROOT

def test_mcp_stdout_remains_json_during_noisy_model_failure(project):
    modules=project.parent/'noisy_modules'
    modules.mkdir()
    (modules/'sentence_transformers.py').write_text(
        'import os\n'
        'class SentenceTransformer:\n'
        ' def __init__(self,*args,**kwargs):\n'
        '  print("HF_PROGRESS_NOISE",flush=True)\n'
        '  os.write(1,b"NATIVE_ERROR_NOISE\\n")\n'
        '  raise RuntimeError("injected model load error")\n',encoding='utf-8')
    env=dict(os.environ,PYTHONPATH=str(modules),PYTHONUNBUFFERED='1')
    diagnostics=project.parent/'stderr.txt'
    with diagnostics.open('w',encoding='utf-8') as error:
        child=subprocess.Popen([sys.executable,str(APP_ROOT/'server.py'),'--allowed-dir',str(project)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error,text=True,encoding='utf-8',env=env)
        lines=queue.Queue()
        reader=threading.Thread(target=lambda:[lines.put(line) for line in child.stdout],daemon=True)
        reader.start()
        observed=[]
        def send(value):
            child.stdin.write(json.dumps(value)+'\n');child.stdin.flush()
        def response(rid):
            while True:
                line=lines.get(timeout=45)
                value=json.loads(line)  # Any stdout progress/error line fails this test.
                observed.append(value)
                if value.get('id')==rid:return value
        try:
            send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{
                'protocolVersion':'2025-11-25','capabilities':{},'clientInfo':{'name':'raw-probe','version':'1'}}})
            assert 'result' in response(1)
            send({'jsonrpc':'2.0','method':'notifications/initialized'})
            send({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{
                'name':'rag_search','arguments':{'query':'calculator'}}})
            result=response(2)
            payload=json.loads(result['result']['content'][0]['text'])
            assert payload['index_available'] is False
            assert payload['retrieval_mode']=='source_fallback'
            send({'jsonrpc':'2.0','id':3,'method':'tools/list'})
            assert 'tools' in response(3)['result']  # Protocol still usable after failure.
        finally:
            child.stdin.close()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            reader.join(timeout=5)
        text=diagnostics.read_text(encoding='utf-8')
        assert 'HF_PROGRESS_NOISE' in text and 'NATIVE_ERROR_NOISE' in text
        assert all(value['jsonrpc']=='2.0' for value in observed)
