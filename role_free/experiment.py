"""Matched small-only, impression-weighted, end-to-end relation experiment."""
import argparse, hashlib, json, math, os, random, time, traceback
import numpy as np
import torch
from .config import ROOT,DATA_ROOT,CFG,RUN_ORDER,MODES
from .data import Impressions
from .model import RoleFreeRanker,training_loss,potential_residual_ratio
from io_utils import read,save,sha,now
from .evaluation import evaluate,progress


def seed_all():
    random.seed(CFG['seed']);np.random.seed(CFG['seed']);torch.manual_seed(CFG['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(CFG['seed'])


def new_model(vocab,mode):
    return RoleFreeRanker(vocab,mode,**{k:CFG[k] for k in
        ('relation_dim','heads','rounds','pair_chunk','news_chunk')})


def state_hash(net,backbone=False):
    from nrms import NRMS
    keys=set(NRMS(2).state_dict()) if backbone else None
    digest=hashlib.sha256()
    for name,value in net.state_dict().items():
        if keys is not None and name not in keys:continue
        digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def sample_columns(data,qq,rng):
    """All positives once per impression; negatives only from the same exposure."""
    owners=[];columns=[]
    for j,q in enumerate(qq):
        lo,hi=map(int,data.a['offsets'][q:q+2]);y=data.a['train_labels'][lo:hi]
        positives=np.flatnonzero(y>0);negatives=np.flatnonzero(y==0)
        if not len(positives) or not len(negatives):
            raise ValueError(f'Unrankable FIT impression {int(q)}')
        for p in positives:
            owners.append(j)
            columns.append(np.r_[p,rng.choice(negatives,CFG['sampled_negatives'],replace=len(negatives)<CFG['sampled_negatives'])])
    return np.asarray(owners,dtype=np.int64),np.asarray(columns,dtype=np.int64)


def update_digest(digest,qq,owners,columns):
    for x in (qq,owners,columns):digest.update(np.asarray(x,dtype=np.int64).tobytes())


def step(net,data,titles,qq,owners,columns,optimizer):
    optimizer.zero_grad(set_to_none=True);start=0;totals=dict(total=0.,ce=0.,relation=0.,pair=0.)
    for micro in data.microbatches(qq):
        h,c,mask=data.inputs(micro,titles.device)
        take=(owners>=start)&(owners<start+len(micro))
        oo=torch.tensor(owners[take]-start,device=titles.device)
        cc=torch.tensor(columns[take],device=titles.device)
        labels=data.fit_labels(micro,c.shape[1],titles.device)
        assert (labels[oo,cc[:,0]]>0).all() and (labels[oo[:,None],cc[:,1:]]==0).all()
        with torch.autocast('cuda',dtype=torch.bfloat16):out=net(titles,h,c,mask)
        loss,parts=training_loss(out,labels,oo,cc,net.mode,CFG['relation_weight'],CFG['pair_weight'])
        assert torch.isfinite(loss)
        weight=len(micro)/len(qq);(loss*weight).backward()
        totals['total']+=float(loss.detach())*weight
        for k,v in parts.items():totals[k]+=float(v.detach())*weight
        start+=len(micro)
    assert start==len(qq)
    norm=torch.nn.utils.clip_grad_norm_(net.parameters(),CFG['gradient_clip'],error_if_nonfinite=True)
    optimizer.step();totals['gradient_norm']=float(norm);return totals


def verify_frozen():
    f=read(ROOT/'FROZEN_PROTOCOL.json');assert f['config']==CFG and f['run_order']==list(RUN_ORDER)
    assert str((DATA_ROOT/'small').resolve())==f['prepared_directory'], 'Prepared data location changed after freeze'
    for path,expected in f['source_sha256'].items():assert sha(path)==expected,path
    for path,expected in f['data_sha256'].items():assert sha(path)==expected,path
    return f


def train(mode):
    frozen=verify_frozen();run=ROOT/'runs'/mode
    if (run/'TRAINING_COMPLETE.json').exists():return
    if (run/'STARTED.json').exists():raise RuntimeError('Incomplete run preserved; explicit recovery required')
    seed_all();data=Impressions('small');titles=torch.tensor(np.array(data.a['titles'],dtype=np.int64),device='cuda')
    net=new_model(data.info['vocab_size'],mode).cuda()
    full=state_hash(net);backbone=state_hash(net,True)
    assert full==frozen['initial_full_sha256'] and backbone==frozen['initial_backbone_sha256']
    # Hashing a reference NRMS creates parameters; reset RNG before training for all modes.
    seed_all();optimizer=torch.optim.Adam(net.parameters(),lr=CFG['lr'])
    save(run/'STARTED.json',dict(time=now(),pid=os.getpid(),mode=mode,config=CFG,
        initial_full_sha256=full,initial_backbone_sha256=backbone,
        parameters=sum(p.numel() for p in net.parameters()),previous_weights_loaded=False,
        fit_impressions=len(data.fit),current_labels_in_forward=False))
    records=[];start_all=time.perf_counter()
    for epoch in range(1,CFG['epochs']+1):
        net.train();rng=np.random.default_rng(CFG['seed']+epoch);order=rng.permutation(data.fit)
        digest=hashlib.sha256();sums=dict(total=0.,ce=0.,relation=0.,pair=0.);seen=0
        torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for batch,st in enumerate(range(0,len(order),CFG['batch_size'])):
            qq=order[st:st+CFG['batch_size']];owners,columns=sample_columns(data,qq,rng)
            update_digest(digest,qq,owners,columns)
            parts=step(net,data,titles,qq,owners,columns,optimizer)
            for k in sums:sums[k]+=parts[k]*len(qq)
            seen+=len(qq)
            if batch%50==0:
                progress(run,state='TRAINING',mode=mode,epoch=epoch,epochs=CFG['epochs'],batch=batch,
                    batches=math.ceil(len(order)/CFG['batch_size']),**{k:v/seen for k,v in sums.items()})
                if (ROOT/'STOP_REQUEST.json').exists():raise RuntimeError('USER_STOP_REQUEST')
        torch.cuda.synchronize();seconds=time.perf_counter()-started;memory=torch.cuda.max_memory_allocated()/2**20
        assert digest.hexdigest()==frozen['expected_sampling_sha256'][str(epoch)]
        checkpoint=run/f'epoch{epoch:02d}.pt'
        assert not checkpoint.exists();torch.save(dict(model=net.state_dict(),mode=mode,epoch=epoch,config=CFG),checkpoint)
        tmp=run/'last_optimizer_state.pt.tmp'
        torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),mode=mode,epoch=epoch,
            config=CFG,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),tmp)
        os.replace(tmp,run/'last_optimizer_state.pt')
        val=evaluate(net,data,titles,'valid',run/f'valid_epoch{epoch:02d}',run)
        entry=dict(epoch=epoch,train_losses={k:v/seen for k,v in sums.items()},train_seconds=seconds,
            peak_cuda_MiB=memory,validation_ndcg10=val['ndcg10'],sampling_sha256=digest.hexdigest(),
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
        records.append(entry);save(run/'EPOCHS.json',records)
        selected=max(records,key=lambda r:(r['validation_ndcg10'],-r['epoch']))
        save(run/'SELECTED_ON_VALID.json',selected);print(json.dumps(entry),flush=True)
    save(run/'TRAINING_COMPLETE.json',dict(time=now(),selected=selected,config=CFG,seed=CFG['seed'],fits=1,
        seconds=time.perf_counter()-start_all,initial_full_sha256=full,initial_backbone_sha256=backbone,
        sampling_hashes={str(r['epoch']):r['sampling_sha256'] for r in records},test_used_for_selection=False))


@torch.inference_mode()
def relation_diagnostics(net,data,titles,run):
    out=run/'VALID_RELATION_DIAGNOSTICS.json'
    if out.exists():return
    net.eval();origins=data.valid[np.linspace(0,len(data.valid)-1,128,dtype=int)]
    rows=[]
    for micro in data.microbatches(origins):
        h,c,mask=data.inputs(micro,'cuda');pred=net(titles,h,c,mask)
        labels=data.fit_labels(micro,c.shape[1],'cuda')
        ratio,energy=potential_residual_ratio(pred['edge_logits'],mask)
        for j,q in enumerate(micro):
            pn=(mask[j]&(labels[j]>0))[:,None]&(mask[j]&(labels[j]==0))[None,:]
            rho=pred['edge_logits'][j][pn]
            rows.append(dict(origin=int(q),pair_count=int(pn.sum()),edge_energy=float(energy[j]),
                potential_residual_ratio=float(ratio[j]) if energy[j]>1e-10 else None,
                observed_pair_direction_accuracy=float((rho>0).float().mean()) if len(rho) else None,
                correction_mean_abs=float(pred['relation_delta'][j][mask[j]].abs().mean())))
    eligible=[r['potential_residual_ratio'] for r in rows if r['potential_residual_ratio'] is not None]
    save(out,dict(time=now(),split='VALID',selection_use=False,rows=rows,
        mean_potential_residual_ratio=float(np.mean(eligible)) if eligible else None,
        caution='Nonzero non-potential energy may be noise; it does not prove useful relations or novelty.'))


def test(mode):
    verify_frozen();freeze_path=ROOT/'ALL_MODELS_FROZEN.json';f=read(freeze_path)
    assert set(f['models'])==set(RUN_ORDER) and not f['test_used_for_selection']
    entry=f['models'][mode];assert sha(entry['checkpoint'])==entry['checkpoint_sha256']
    seed_all();data=Impressions('small');net=new_model(data.info['vocab_size'],mode).cuda()
    net.load_state_dict(torch.load(entry['checkpoint'],map_location='cuda',weights_only=False)['model']);net.requires_grad_(False)
    titles=torch.tensor(np.array(data.a['titles'],dtype=np.int64),device='cuda');run=ROOT/'runs'/mode
    relation_diagnostics(net,data,titles,run)
    result=evaluate(net,data,titles,'test',run/'research_TEST',run,freeze_path)
    progress(run,state='TEST_COMPLETE',mode=mode,selected_epoch=entry['epoch'],**{k:result[k] for k in ('auc','mrr','ndcg5','ndcg10')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--test',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    try:test(a.mode) if a.test else train(a.mode)
    except BaseException:
        save(ROOT/'runs'/a.mode/('TEST_ERROR.json' if a.test else 'ERROR.json'),dict(time=now(),traceback=traceback.format_exc()));raise
