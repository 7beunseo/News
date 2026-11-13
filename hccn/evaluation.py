"""Unchanged whole-candidate evaluator from the frozen RCRIN comparison."""
from pathlib import Path
import os,time,csv,gc
import numpy as np
import torch
import metrics
from metrics import independent_metrics as independent
from io_utils import read,save,sha,now
KEYS=('auc','mrr','ndcg5','ndcg10')

def progress(run,**values):
    values.update(time=now(),pid=os.getpid());save(run/'PROGRESS.json',values)
    print(__import__('json').dumps(values,ensure_ascii=False),flush=True)


@torch.inference_mode()
def evaluate(model,data,titles,split,out,run,freeze=None):
    if (out/'COMPLETE.json').exists():return read(out/'COMPLETE.json')
    if (out/'STARTED.json').exists():raise RuntimeError('Incomplete evaluation preserved: '+str(out))
    save(out/'STARTED.json',dict(time=now(),split=split,whole_candidate_context=True));model.eval()
    origins=data.origins(split);offsets=np.r_[0,np.cumsum(data.lengths[origins],dtype=np.int64)]
    np.save(out/'origins.npy',origins);np.save(out/'offsets.npy',offsets)
    cache=torch.zeros(len(titles),256,device=titles.device);start=time.perf_counter()
    for lo in range(1,len(titles),128):cache[lo:lo+128]=model.encode_news(titles[lo:lo+128])
    scores=np.lib.format.open_memmap(out/'scores.npy',mode='w+',dtype='float32',shape=(int(offsets[-1]),))
    done=0;parity=0.;next_print=0;correction_abs=0.;candidates=0
    for st in range(0,len(origins),32):
        for qq in data.microbatches(origins[st:st+32]):
            h,c,mask=data.inputs(qq,titles.device)
            output=model.cached(cache,h,c,mask)
            value=output['scores'];assert torch.isfinite(value[mask]).all()
            if done==0:
                raw=model(titles,h[:1],c[:1],mask[:1])['scores']
                parity=float((raw-value[:1]).abs().max())
                assert torch.allclose(raw,value[:1],atol=4e-4,rtol=1e-4),(parity,model.mode)
            v=value.cpu().numpy()
            for j,q in enumerate(qq):scores[offsets[done+j]:offsets[done+j+1]]=v[j,:int(data.lengths[q])]
            if 'relation_delta' in output:correction_abs+=float(output['relation_delta'][mask].abs().sum())
            candidates+=int(mask.sum());done+=len(qq)
            if done>=next_print:
                progress(run,state='SCORING',split=split,model=model.mode,done=done,total=len(origins))
                next_print=done+5000
    scores.flush();del scores,cache;torch.cuda.synchronize();inference_seconds=time.perf_counter()-start
    score_hash=sha(out/'scores.npy')
    # The complete score file exists before any current TEST outcome is read.
    labels,label_start=data.evaluation_labels(split,freeze)
    scores=np.load(out/'scores.npy',mmap_mode='r');sums=np.zeros(4);eligible=excluded=0;audit_error=0.;sizes={}
    with (out/'SESSIONS.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['origin','impression','user','candidates',*KEYS]);writer.writeheader()
        for j,q in enumerate(origins):
            lo=int(data.a['offsets'][q])-label_start;hi=int(data.a['offsets'][q+1])-label_start
            y=labels[lo:hi];s=scores[offsets[j]:offsets[j+1]];m=metrics.metric(y,s)
            if m is None:excluded+=1;continue
            if j%1000==0:
                check=independent(y,s);audit_error=max(audit_error,max(abs(m[k]-check[k]) for k in KEYS))
            v=np.array([m[k] for k in KEYS]);sums+=v;eligible+=1
            group='2-10' if len(y)<=10 else ('11-50' if len(y)<=50 else '51+')
            if group not in sizes:sizes[group]=dict(origins=0,total=np.zeros(4))
            sizes[group]['origins']+=1;sizes[group]['total']+=v
            writer.writerow(dict(origin=int(q),impression=int(data.a['impressions'][q]),user=int(data.a['users'][q]),candidates=len(y),**m))
    assert eligible>0 and audit_error<1e-10
    result=dict(time=now(),dataset=data.dataset,mode=model.mode,split=split,origins=len(origins),eligible_origins=eligible,
        excluded_degenerate=excluded,candidates=int(offsets[-1]),inference_seconds=inference_seconds,
        cached_raw_max_error=parity,independent_metric_max_error=audit_error,score_sha256=score_hash,
        **dict(zip(KEYS,map(float,sums/eligible))))
    result['candidate_size_groups']={k:dict(origins=v['origins'],**dict(zip(KEYS,map(float,v['total']/v['origins'])))) for k,v in sizes.items()}
    result['test_definition']='official labelled DEV full; reused internal research TEST, not challenge TEST'
    save(out/'COMPLETE.json',result);gc.collect();torch.cuda.empty_cache();return result
