"""Standalone small-only queue; no dependency on another machine's processes."""
import os,subprocess,sys,time,traceback
import psutil
from .config import ROOT,REPO,CFG,RUN_ORDER
from io_utils import read,save,now
from .experiment import verify_frozen
from .report import report


def state(**values):
    values.update(time=now(),pid=os.getpid());save(ROOT/'PROGRESS.json',values);report()
    print(__import__('json').dumps(values,ensure_ascii=False),flush=True)


def stage(mode,test=False):
    verify_frozen()
    if (ROOT/'STOP_REQUEST.json').exists():raise RuntimeError('USER_STOP_REQUEST')
    if psutil.disk_usage(str(ROOT)).free<10*2**30:raise RuntimeError('Insufficient free disk; results preserved')
    command=[sys.executable,'-u','-B','-X','utf8','-m','role_free.experiment','--mode',mode]
    if test:command.append('--test')
    log=ROOT/'logs'/(mode+('_test.log' if test else '_train.log'));log.parent.mkdir(exist_ok=True)
    with log.open('x',encoding='utf-8') as output:
        p=subprocess.Popen(command,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            encoding='utf-8',errors='replace',bufsize=1)
        state(state='TESTING' if test else 'TRAINING',mode=mode,worker_pid=p.pid,epochs=CFG['epochs'])
        save(ROOT/'CURRENT_STAGE.json',dict(time=now(),mode=mode,test=test,pid=p.pid,create_time=psutil.Process(p.pid).create_time()))
        for line in p.stdout:
            output.write(line);output.flush();print(line,end='',flush=True)
        if p.wait()!=0:raise RuntimeError('Worker failed; preserve checkpoint/log: '+str(log))
    report()


def main():
    verify_frozen()
    if (ROOT/'QUEUE_STARTED.json').exists():raise RuntimeError('Existing queue preserved')
    save(ROOT/'QUEUE_STARTED.json',dict(time=now(),pid=os.getpid(),create_time=psutil.Process().create_time(),order=RUN_ORDER,config=CFG))
    selected={};common_hash=None;common_samples=None
    for mode in RUN_ORDER:
        run=ROOT/'runs'/mode
        if not (run/'TRAINING_COMPLETE.json').exists():stage(mode)
        done=read(run/'TRAINING_COMPLETE.json');epochs=read(run/'EPOCHS.json')
        assert [r['epoch'] for r in epochs]==[1,2,3,4,5]
        if common_hash is None:common_hash=done['initial_full_sha256'];common_samples=done['sampling_hashes']
        assert done['initial_full_sha256']==common_hash and done['sampling_hashes']==common_samples
        selected[mode]=done['selected']
    save(ROOT/'ALL_MODELS_FROZEN.json',dict(time=now(),models=selected,test_used_for_selection=False,
        matched_initialization_and_sampling=True,selection='full VALID NDCG@10; earliest tie'))
    for mode in RUN_ORDER:stage(mode,test=True)
    save(ROOT/'COMPLETE.json',dict(time=now(),fits=len(RUN_ORDER),independent_seeds=1,epochs=5,report='RESULT_KO.html'))
    state(state='COMPLETE',fits=len(RUN_ORDER),independent_seeds=1)


if __name__=='__main__':
    try:main()
    except BaseException:
        save(ROOT/'QUEUE_ERROR.json',dict(time=now(),traceback=traceback.format_exc()));state(state='ERROR');raise
