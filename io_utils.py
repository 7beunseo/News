from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,os,time,uuid,hashlib,importlib.util,sys

def now():return datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(200):
        try:os.replace(temp,path);return
        except PermissionError:
            if attempt==199:raise
            time.sleep(.05)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
