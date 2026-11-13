"""Frozen HCCN/ablation queue; one new GPU worker at a time."""
import os,subprocess,sys,traceback
import psutil
from .config import ROOT,REPO,CFG,SEEDS,RUN_ORDER
from .experiment import verify_frozen
from .report import report
from io_utils import read,save,now


def state(**values):
    save(ROOT/'PROGRESS.json',dict(time=now(),pid=os.getpid(),**values));report()
    print(__import__('json').dumps(values),flush=True)


def stage(mode,seed,test=False):
    verify_frozen()
    if (ROOT/'STOP_REQUEST.json').exists():raise RuntimeError('USER_STOP_REQUEST')
    if psutil.disk_usage(str(ROOT)).free<10*2**30:raise RuntimeError('Disk space below 10 GiB; results preserved')
    args=[sys.executable,'-u','-B','-X','utf8','-m','hccn.experiment','--mode',mode,'--seed',str(seed)]
    if test:args.append('--test')
    path=ROOT/'logs'/f'{seed}_{mode}_{"test" if test else "train"}.log';path.parent.mkdir(exist_ok=True)
    with path.open('x',encoding='utf-8') as log:
        p=subprocess.Popen(args,cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        state(state='TESTING' if test else 'TRAINING',mode=mode,seed=seed,worker_pid=p.pid,epochs=5)
        save(ROOT/'CURRENT_STAGE.json',dict(time=now(),mode=mode,seed=seed,test=test,pid=p.pid,create_time=psutil.Process(p.pid).create_time()))
        for line in p.stdout:log.write(line);log.flush();print(line,end='',flush=True)
        if p.wait()!=0:raise RuntimeError('Worker failed; checkpoint and logs preserved: '+str(path))
    report()


def main():
    verify_frozen()
    if (ROOT/'QUEUE_STARTED.json').exists():raise RuntimeError('Existing queue preserved')
    save(ROOT/'QUEUE_STARTED.json',dict(time=now(),pid=os.getpid(),create_time=psutil.Process().create_time(),
        seeds=SEEDS,order=RUN_ORDER,planned_fits=len(SEEDS)*len(RUN_ORDER)))
    for seed in SEEDS:
        selected={};shared=None;samples=None
        for mode in RUN_ORDER:
            run=ROOT/'runs'/str(seed)/mode
            if not (run/'TRAINING_COMPLETE.json').exists():stage(mode,seed)
            done=read(run/'TRAINING_COMPLETE.json');records=read(run/'EPOCHS.json')
            assert [r['epoch'] for r in records]==[1,2,3,4,5]
            if samples is None:samples=done['sampling_hashes']
            assert samples==done['sampling_hashes']
            if mode!='nrms_ce':
                if shared is None:shared=done['initial_full_sha256']
                assert shared==done['initial_full_sha256']
            selected[mode]=done['selected']
        save(ROOT/'frozen'/f'{seed}.json',dict(time=now(),seed=seed,models=selected,test_used_for_selection=False,
            settings_frozen_before_any_test=True,selection='VALID NDCG@10, epochs 1-5'))
        for mode in RUN_ORDER:stage(mode,seed,test=True)
    save(ROOT/'COMPLETE.json',dict(time=now(),fits=18,independent_seeds=3,report='RESULT_KO.html'))
    state(state='COMPLETE',fits=18,independent_seeds=3)


if __name__=='__main__':
    try:main()
    except BaseException:
        save(ROOT/'QUEUE_ERROR.json',dict(time=now(),traceback=traceback.format_exc()));state(state='ERROR');raise
