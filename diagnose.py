"""VALID score intervention and FIT gradient diagnostics; never opens TEST labels."""
import csv,json,math,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from config import ROOT,CFG
from data import Impressions
from io_utils import now,read,save,sha
from train import new_model,seed_all,metrics,independent,KEYS
from model import reciprocal_training_loss


def describe(values):
    x=np.asarray(values,dtype=np.float64)
    if not len(x):return dict(count=0)
    return dict(count=int(x.size),mean=float(x.mean()),mean_abs=float(np.abs(x).mean()),std=float(x.std()),
                quantiles=dict(zip(['min','p01','p10','median','p90','p99','max'],map(float,np.quantile(x,[0,.01,.1,.5,.9,.99,1])))))


def imbalance(data):
    n=data.base_info['splits']['train']['origins'];offsets=data.a['offsets'][:n]
    pos=np.add.reduceat(data.a['train_labels'],offsets,dtype=np.int64)
    result={}
    for name,qq in [('FIT',data.fit),('VALID',data.valid)]:
        count=data.lengths[qq];p=pos[qq];neg=count-p
        result[name]=dict(origins=len(qq),candidates=int(count.sum()),clicked_candidates=int(p.sum()),
            nonclicked_candidates=int(neg.sum()),candidate_click_fraction=float(p.sum()/count.sum()),
            mean_impression_click_fraction=float(np.mean(p/count)),
            no_click_origins=int((p==0).sum()),all_click_origins=int((neg==0).sum()),
            no_click_fraction=float(np.mean(p==0)),all_click_fraction=float(np.mean(neg==0)),
            ranking_eligible_origins=int(((p>0)&(neg>0)).sum()))
    return result


@torch.inference_mode()
def score_intervention(net,data,titles,out,selected):
    origins=data.valid;offsets=np.r_[0,np.cumsum(data.lengths[origins],dtype=np.int64)]
    np.save(out/'origins.npy',origins);np.save(out/'offsets.npy',offsets)
    array=np.lib.format.open_memmap(out/'valid_diagnostics.npy',mode='w+',dtype='float32',shape=(int(offsets[-1]),12))
    columns=['base','relation','full','role_probability','mass_plus_round1','mass_minus_round1',
             'mass_plus_round2','mass_minus_round2','gate_mean','delta_plus_rms','delta_minus_rms','click_label_VALID_only']
    save(out/'COLUMNS.json',columns)
    cache=torch.zeros(len(titles),256,device=titles.device)
    for lo in range(1,len(titles),128):cache[lo:lo+128]=net.encode_news(titles[lo:lo+128])
    gate={}
    handle=net.relation.fusion_gate.register_forward_hook(lambda module,args,result:gate.update(value=result.sigmoid().mean(-1).detach()))
    rows=[];done=0;next_print=0;audit_error=0.;historical_error=0.
    reference=np.load(Path(selected['checkpoint']).parent/f"valid_epoch{selected['epoch']:02d}"/'scores.npy',mmap_mode='r')
    bce_pos=bce_neg=0.;eligible=0;centers=[];pair_flips=pair_total=0
    for start in range(0,len(origins),32):
        for qq in data.microbatches(origins[start:start+32]):
            h,c,mask=data.inputs(qq,'cuda');output=net.cached(cache,h,c,mask,True)
            y=data.fit_labels(qq,c.shape[1],'cuda')  # outcomes only after forward
            features=torch.stack([output['base_logits'],output['relation_delta'],output['scores'],output['role_prob'],
                output['role_mass_plus'][0],output['role_mass_minus'][0],output['role_mass_plus'][1],output['role_mass_minus'][1],
                gate['value'],output['delta_plus'].square().mean(-1).sqrt(),output['delta_minus'].square().mean(-1).sqrt(),y],-1).float().cpu().numpy()
            for j,q in enumerate(qq):
                length=int(data.lengths[q]);v=features[j,:length];lo,hi=map(int,offsets[done+j:done+j+2]);array[lo:hi]=v
                historical_error=max(historical_error,float(np.max(np.abs(v[:,2]-reference[lo:hi]))))
                base,full,label=v[:,0],v[:,2],v[:,-1];a,b=metrics.metric(label,base),metrics.metric(label,full)
                bce_pos+=float(np.logaddexp(0,-full[label>0]).sum()/length)
                bce_neg+=float(np.logaddexp(0,full[label<=0]).sum()/length)
                centers.append([float(np.std(base)),float(np.std(v[:,1])),float(np.std(full))])
                if a is None:continue
                eligible+=1;row=dict(origin=int(q),user=int(data.a['users'][q]),candidates=length,
                    clicks=int(label.sum()),top1_changed=bool(np.argmax(base)!=np.argmax(full)))
                for k in KEYS:row['base_'+k]=a[k];row['full_'+k]=b[k];row['delta_'+k]=b[k]-a[k]
                rows.append(row)
                bd=base[label>0,None]-base[None,label<=0];fd=full[label>0,None]-full[None,label<=0]
                pair_flips+=int(np.count_nonzero(np.sign(bd)!=np.sign(fd)));pair_total+=bd.size
                if (done+j)%500==0:
                    for score,res in [(base,a),(full,b)]:
                        independent_result=independent(label,score)
                        audit_error=max(audit_error,max(abs(res[k]-independent_result[k]) for k in KEYS))
            done+=len(qq)
            if done>=next_print:
                save(out/'PROGRESS.json',dict(time=now(),state='VALID_INTERVENTION',done=done,total=len(origins)))
                print(f'VALID {done}/{len(origins)}',flush=True);next_print=done+3000
    handle.remove();array.flush();assert audit_error<1e-10 and historical_error<5e-4,(audit_error,historical_error)
    with (out/'PER_ORIGIN.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stats={}
    for name,group in [('all',np.ones(len(array),bool)),('clicked',array[:,-1]>0),('nonclicked',array[:,-1]<=0)]:
        stats[name]={key:describe(array[group,k]) for k,key in enumerate(columns[:-1])}
    users=np.array([r['user'] for r in rows]);_,ix=np.unique(users,return_inverse=True)
    counts=np.bincount(ix);values=np.array([r['delta_ndcg10'] for r in rows]);sums=np.bincount(ix,weights=values)
    rng=np.random.default_rng(26111921);boot=[]
    for _ in range(1000):
        sample=rng.integers(len(counts),size=len(counts));boot.append(float(sums[sample].sum()/counts[sample].sum()))
    result=dict(time=now(),checkpoint_epoch=selected['epoch'],split='VALID',eligible_origins=eligible,
        base_only={k:float(np.mean([r['base_'+k] for r in rows])) for k in KEYS},
        full={k:float(np.mean([r['full_'+k] for r in rows])) for k in KEYS},
        delta_full_minus_base={k:float(np.mean([r['delta_'+k] for r in rows])) for k in KEYS},
        paired_user_interval_ndcg10=list(map(float,np.quantile(boot,[.025,.975]))),bootstrap='1000 paired user clusters; descriptive VALID-selected snapshot',
        top1_changed_fraction=float(np.mean([r['top1_changed'] for r in rows])),positive_negative_pair_order_changed_fraction=pair_flips/max(1,pair_total),
        mean_within_impression_score_std=dict(zip(['base','relation','full'],map(float,np.mean(centers,0)))),
        bce_positive_contribution=bce_pos/len(origins),bce_negative_contribution=bce_neg/len(origins),
        beta=float(F.softplus(net.relation.beta_raw)),distributions=stats,
        numerical_audit_error=audit_error,historical_full_score_max_error=historical_error,
        intervention='same trained weights; relation correction set to zero only; not a from-scratch ablation',test_labels_opened=False)
    save(out/'COMPLETE.json',result);del cache,array;torch.cuda.empty_cache();return result


def gradient_audit(data,titles,epochs,out,mode):
    # Fixed FIT sample, no outcome-based sample selection or parameter updates.
    sample_size=min(16,len(data.fit));assert sample_size>0
    sample=np.random.default_rng(26111922).choice(data.fit,sample_size,replace=False)
    np.save(out/'FIT_SAMPLE.npy',sample);rows=[]
    for record in epochs:
        assert sha(record['checkpoint'])==record['checkpoint_sha256']
        net=new_model(data.info['vocab_size'],mode).cuda()
        state=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
        net.load_state_dict(state['model']);del state
        net.train();params=[p for n,p in net.named_parameters() if n.startswith(('words.','word_attention.','word_pool.'))]
        for i in range(0,len(sample),4):
            seed_all(26111923+i);qq=sample[i:i+4];h,c,mask=data.inputs(qq,'cuda');y=data.fit_labels(qq,c.shape[1],'cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):prediction=net(titles,h,c,mask)
            total,parts=reciprocal_training_loss(prediction,y,mode,CFG['role_weight'],CFG['edge_weight'],CFG['pair_weight'])
            gradients={}
            for j,key in enumerate(('click','role','edge','pair')):
                g=torch.autograd.grad(parts[key],params,retain_graph=j<3,allow_unused=True)
                gradients[key]=torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1).float() for p,v in zip(params,g)])
            aux=CFG['role_weight']*gradients['role']+CFG['edge_weight']*gradients['edge']+CFG['pair_weight']*gradients['pair']
            gradients['weighted_aux']=aux
            def cos(x,y):
                denom=x.norm()*y.norm();return float(torch.dot(x,y)/denom) if float(denom)>0 else None
            rows.append(dict(epoch=record['epoch'],fit_batch=i//4,origins=list(map(int,qq)),
                losses={k:float(v.detach()) for k,v in parts.items()},
                gradient_norms={k:float(v.norm()) for k,v in gradients.items()},
                cosine_with_click={k:cos(gradients['click'],v) for k,v in gradients.items() if k!='click'}))
            del gradients,aux,prediction,parts,total,g
        del net;torch.cuda.empty_cache();print(f'FIT gradient audit epoch {record["epoch"]} complete',flush=True)
    result=dict(rows=rows,fit_origins=sample_size,batches_per_epoch=math.ceil(sample_size/4),optimizer_steps=0,
        parameters='shared news encoder: words, word_attention, word_pool',mode='train mode, fixed dropout seed, BF16 forward',
        interpretation='small diagnostic sample, not proof of a training-wide causal conflict',test_labels_opened=False)
    save(out/'GRADIENTS.json',result);return result


def execute(args, out):
    from train import verify_protocol, run_path
    verify_protocol()
    assert CFG['rounds'] == 2, 'The diagnostic columns describe exactly two exchange rounds.'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    run=run_path(args.dataset,args.seed,args.mode)
    records=[r for r in read(run/'EPOCHS.json') if 1<=r['epoch']<=CFG['epochs']]
    assert records and CFG['epochs']<=5
    selected=max(records,key=lambda r:(r['validation_ndcg10'],-r['epoch']))
    assert sha(selected['checkpoint'])==selected['checkpoint_sha256']
    save(out/'FROZEN_SELECTION.json',dict(time=now(),dataset=args.dataset,mode=args.mode,seed=args.seed,
        selected=selected,epochs=records,test_used_for_selection=False,test_labels_opened=False,
        source_sha256=sha(__file__),fit_complete=(run/'TRAINING_COMPLETE.json').exists()))
    data=Impressions(args.dataset)
    assert data.a['timestamps'][data.fit].max()<data.a['timestamps'][data.valid].min()
    counts=imbalance(data);save(out/'IMBALANCE.json',counts)
    seed_all(args.seed);net=new_model(data.info['vocab_size'],args.mode).cuda()
    state=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    assert state['mode']==args.mode and state['seed']==args.seed and state['epoch']==selected['epoch']
    net.load_state_dict(state['model']);del state;net.eval()
    titles=torch.tensor(np.array(data.a['titles'],dtype=np.int64),device='cuda')
    intervention=out/'intervention';intervention.mkdir()
    scores=score_intervention(net,data,titles,intervention,selected)
    del net;torch.cuda.empty_cache()
    if not args.skip_gradients:gradient_audit(data,titles,records,out,args.mode)
    save(out/'EPOCH_LOSSES.json',dict(epochs=records,
        note='loss_components are saved epoch means where available; missing old fields remain missing.'))
    assert sha(selected['checkpoint'])==selected['checkpoint_sha256']
    verify_protocol()
    summary=dict(time=now(),state='COMPLETE',dataset=args.dataset,mode=args.mode,
        selected_epoch=selected['epoch'],base_only=scores['base_only'],full=scores['full'],
        delta_full_minus_base=scores['delta_full_minus_base'],gradient_audit_run=not args.skip_gradients,
        test_labels_opened=False,optimizer_steps=0,
        limitation='VALID-selected same-weight intervention and a small FIT gradient sample; not independent confirmation or a training-wide causal conclusion.')
    save(out/'COMPLETE.json',summary)
    save(out/'PROGRESS.json',dict(time=now(),state='COMPLETE'))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    import argparse
    from model import MODES
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['small','large'],required=True)
    parser.add_argument('--mode',choices=[m for m in MODES if not m.startswith('nrms_')],default='rcrin_full')
    parser.add_argument('--seed',type=int,default=26111801)
    parser.add_argument('--skip-gradients',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    assert out.is_relative_to(ROOT/'diagnostics'),'Keep generated outputs under diagnostics/.'
    out.mkdir(parents=True,exist_ok=False)
    try:execute(args,out)
    except BaseException:
        save(out/'ERROR.json',dict(time=now(),traceback=traceback.format_exc()));raise
