from pathlib import Path
import os,sys,random,time,csv,argparse,traceback,hashlib,gc
import numpy as np
import torch
from config import ROOT,CFG,SEEDS,RUN_ORDER
from model import RCRINRanker,MODES
from model import reciprocal_training_loss
from data import Impressions
from io_utils import now,read,save,sha,load_module

import metrics
from metrics import independent_metrics as independent
KEYS=('auc','mrr','ndcg5','ndcg10')

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def new_model(vocab,mode):
    return RCRINRanker(vocab,mode,**{k:CFG[k] for k in ('relation_dim','heads','rounds','pair_chunk','query_chunk','news_chunk','pair_checkpoint')})
def run_path(dataset,seed,mode):return ROOT/'runs'/dataset/str(seed)/mode
def progress(run,**values):
    values.update(time=now(),pid=os.getpid());save(run/'PROGRESS.json',values)
    print(__import__('json').dumps(values,ensure_ascii=False),flush=True)
def verify_protocol():
    frozen=read(ROOT/'FROZEN_PROTOCOL.json');assert frozen['config']==CFG
    for p,h in frozen['source_hashes'].items():assert sha(ROOT/p)==h,p


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
        correction_mean_abs=correction_abs/max(1,candidates),**dict(zip(KEYS,map(float,sums/eligible))))
    result['candidate_size_groups']={k:dict(origins=v['origins'],**dict(zip(KEYS,map(float,v['total']/v['origins'])))) for k,v in sizes.items()}
    result['test_definition']='official labelled DEV full; reused internal research TEST, not challenge TEST'
    save(out/'COMPLETE.json',result);gc.collect();torch.cuda.empty_cache();return result


def train(dataset,mode,seed):
    run=run_path(dataset,seed,mode)
    if (run/'TRAINING_COMPLETE.json').exists():return
    if (run/'STARTED.json').exists():raise RuntimeError('Existing incomplete run preserved')
    verify_protocol();seed_all(seed);data=Impressions(dataset)
    titles=torch.tensor(np.array(data.a['titles'],dtype=np.int64),device='cuda')
    model=new_model(data.info['vocab_size'],mode).cuda();opt=torch.optim.Adam(model.parameters(),lr=CFG['lr'])
    backbone_hash=hashlib.sha256()
    for k,v in model.state_dict().items():
        if not k.startswith('relation.'):backbone_hash.update(k.encode());backbone_hash.update(v.detach().cpu().numpy().tobytes())
    save(run/'STARTED.json',dict(time=now(),pid=os.getpid(),dataset=dataset,mode=mode,seed=seed,config=CFG,
        backbone_initial_sha256=backbone_hash.hexdigest(),parameters=sum(p.numel() for p in model.parameters()),
        backbone_parameters=sum(p.numel() for n,p in model.named_parameters() if not n.startswith('relation.')),
        previous_weights_loaded=False,fit_origins=len(data.fit),full_candidate_count=int(data.lengths[data.fit].sum())))
    best=None;records=[];samplers={};start_all=time.perf_counter()
    for epoch in range(1,CFG['epochs']+1):
        model.train();seed_all(seed+epoch*100003);rng=np.random.default_rng(seed+epoch)
        order=rng.permutation(data.fit);order_hash=hashlib.sha256(order.tobytes()).hexdigest();summed=0.;terms=np.zeros(4);seen=0
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();batches=(len(order)+CFG['batch_size']-1)//CFG['batch_size']
        for batch,st in enumerate(range(0,len(order),CFG['batch_size'])):
            qq=order[st:st+CFG['batch_size']];opt.zero_grad(set_to_none=True)
            for micro in data.microbatches(qq):
                h,c,mask=data.inputs(micro,titles.device);y=data.fit_labels(micro,c.shape[1],titles.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model(titles,h,c,mask)
                loss,parts=reciprocal_training_loss(output,y,mode,CFG['role_weight'],CFG['edge_weight'],CFG['pair_weight'])
                assert torch.isfinite(loss);(loss*len(micro)/len(qq)).backward()
                summed+=float(loss.detach())*len(micro);seen+=len(micro)
                terms+=np.array([float(parts[k].detach()) for k in ('click','role','edge','pair')])*len(micro)
            torch.nn.utils.clip_grad_norm_(model.parameters(),CFG['gradient_clip'],error_if_nonfinite=True);opt.step()
            if batch%100==0:
                progress(run,state='TRAINING',dataset=dataset,model=mode,seed=seed,epoch=epoch,epochs=CFG['epochs'],batch=batch,batches=batches,
                         loss=summed/seen,loss_components=dict(zip(('click','role','edge','pair'),map(float,terms/seen))))
                if (ROOT/'STOP_REQUEST.json').exists():raise RuntimeError('USER_STOP_REQUEST: queue stopped; completed files preserved')
        torch.cuda.synchronize();seconds=time.perf_counter()-start;memory=torch.cuda.max_memory_allocated()/2**20
        cp=run/f'epoch{epoch:02d}.pt';torch.save(dict(model=model.state_dict(),epoch=epoch,mode=mode,seed=seed,config=CFG),cp)
        tmp=run/'last_optimizer_state.pt.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,
            mode=mode,seed=seed,config=CFG,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp);os.replace(tmp,run/'last_optimizer_state.pt')
        val=evaluate(model,data,titles,'valid',run/f'valid_epoch{epoch:02d}',run)
        entry=dict(epoch=epoch,training_loss=summed/seen,train_seconds=seconds,peak_cuda_MiB=memory,
            validation_ndcg10=val['ndcg10'],loss_components=dict(zip(('click','role','edge','pair'),map(float,terms/seen))),
            origin_order_sha256=order_hash,checkpoint=str(cp),checkpoint_sha256=sha(cp))
        records.append(entry);samplers[str(epoch)]=order_hash;save(run/'EPOCHS.json',records)
        if best is None or val['ndcg10']>best['validation_ndcg10']:best=entry
        save(run/'SELECTED_ON_VALID.json',best);print(__import__('json').dumps(entry),flush=True)
    save(run/'TRAINING_COMPLETE.json',dict(time=now(),dataset=dataset,mode=mode,seed=seed,checkpoint=best['checkpoint'],
        checkpoint_sha256=best['checkpoint_sha256'],selected_epoch=best['epoch'],validation_ndcg10=best['validation_ndcg10'],
        origin_order_hashes=samplers,backbone_initial_sha256=backbone_hash.hexdigest(),seconds=time.perf_counter()-start_all,
        fits=1,test_used_for_selection=False))


def test(dataset,mode,seed):
    verify_protocol();freeze_path=ROOT/'frozen'/dataset/f'seed_{seed}.json';freeze=read(freeze_path)
    assert set(freeze['models'])==set(RUN_ORDER);selected=freeze['models'][mode]
    assert sha(selected['checkpoint'])==selected['checkpoint_sha256']
    seed_all(seed);data=Impressions(dataset);model=new_model(data.info['vocab_size'],mode).cuda()
    model.load_state_dict(torch.load(selected['checkpoint'],map_location='cuda',weights_only=False)['model']);model.requires_grad_(False)
    titles=torch.tensor(np.array(data.a['titles'],dtype=np.int64),device='cuda');run=run_path(dataset,seed,mode)
    result=evaluate(model,data,titles,'test',run/'research_TEST',run,freeze_path);print(__import__('json').dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['small','large'],required=True)
    parser.add_argument('--mode',choices=MODES,required=True);parser.add_argument('--seed',type=int,required=True);parser.add_argument('--test',action='store_true')
    args=parser.parse_args();run=run_path(args.dataset,args.seed,args.mode)
    try:
        torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        test(args.dataset,args.mode,args.seed) if args.test else train(args.dataset,args.mode,args.seed)
    except BaseException:save(run/('TEST_ERROR.json' if args.test else 'ERROR.json'),dict(time=now(),traceback=traceback.format_exc()));raise
