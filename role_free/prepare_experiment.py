"""Preflight only: no TEST outcomes, no persistent training checkpoint."""
import gc,hashlib,json,os,subprocess,sys,time
import numpy as np
import torch
from .config import ROOT,REPO,CFG,RUN_ORDER
from .data import Impressions
from .experiment import new_model,seed_all,state_hash,sample_columns,update_digest,step
from io_utils import save,sha,now


def main():
    if (ROOT/'FROZEN_PROTOCOL.json').exists():raise RuntimeError('Already frozen; existing protocol preserved')
    logs=ROOT/'preflight';logs.mkdir(exist_ok=True)
    test=subprocess.run([sys.executable,'-u','-B','-X','utf8','-m','unittest','discover','-s',str(REPO/'tests'),'-p','test_role_free.py','-v'],cwd=REPO,
        text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (logs/'UNIT_TESTS.txt').write_text(test.stdout,encoding='utf-8');print(test.stdout,flush=True)
    if test.returncode:raise RuntimeError('Unit tests failed')
    data=Impressions('small');audit=data.audit();print('Causal audit passed; computing five frozen sample streams',flush=True)
    sampling={};positive_counts={}
    for epoch in range(1,6):
        rng=np.random.default_rng(CFG['seed']+epoch);order=rng.permutation(data.fit);digest=hashlib.sha256();count=0
        for st in range(0,len(order),CFG['batch_size']):
            qq=order[st:st+CFG['batch_size']];owners,columns=sample_columns(data,qq,rng)
            update_digest(digest,qq,owners,columns);count+=len(owners)
        sampling[str(epoch)]=digest.hexdigest();positive_counts[str(epoch)]=count
        print(json.dumps(dict(epoch=epoch,positive_terms=count,sampling_sha256=digest.hexdigest())),flush=True)
    seed_all();net=new_model(data.info['vocab_size'],'rf_full');full=state_hash(net);base=state_hash(net,True)
    total_parameters=sum(p.numel() for p in net.parameters());del net;gc.collect()
    assert torch.cuda.is_available();titles=torch.tensor(np.array(data.a['titles'],dtype=np.int64),device='cuda')
    gpu=[]
    # Exercise the largest real FIT exposure and a representative multi-row batch.
    largest=int(data.fit[np.argmax(data.lengths[data.fit])])
    typical=data.fit[np.argsort(np.abs(data.lengths[data.fit]-40),kind='stable')[:8]]
    for tag,qq in [('largest',np.array([largest])),('typical',np.asarray(typical))]:
        seed_all();net=new_model(data.info['vocab_size'],'rf_full').cuda().train()
        opt=torch.optim.Adam(net.parameters(),lr=CFG['lr']);owners,columns=sample_columns(data,qq,np.random.default_rng(21))
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        results=[step(net,data,titles,qq,owners,columns,opt) for _ in range(2)]
        torch.cuda.synchronize()
        gradients={k:sum(float(p.grad.abs().sum()) for p in getattr(net,k).parameters() if p.grad is not None)
            for k in ('words','relation','edge_score','attention','head')}
        assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
        entry=dict(batch=tag,origins=qq.tolist(),candidate_counts=data.lengths[qq].tolist(),steps=2,
            seconds=time.perf_counter()-t,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
            losses=results,gradient_sums=gradients)
        gpu.append(entry);print(json.dumps(entry),flush=True)
        del net,opt;gc.collect();torch.cuda.empty_cache()
    del titles;gc.collect();torch.cuda.empty_cache()
    save(logs/'GPU_SMOKE.json',dict(time=now(),gpu=torch.cuda.get_device_name(),measurements=gpu,
        concurrent_training=None,full_training_speed_claim=False,discarded_smoke_weights=True))
    input_files=[data.prep/(n+'.npy') for n in ('items','offsets','users','timestamps','impressions',
        'train_labels','matured','titles','provided','train_origins','valid_origins')]
    input_files.extend([data.prep/'NEWS_IDS.json',data.prep/'COMPLETE.json'])
    protocol=dict(time=now(),dataset='MIND-small only',config=CFG,run_order=list(RUN_ORDER),
        prepared_directory=str(data.prep.resolve()),
        source_sha256={str(p):sha(p) for p in [*sorted(ROOT.glob('*.py')),REPO/'config.py',REPO/'nrms.py',REPO/'io_utils.py',REPO/'metrics.py',REPO/'tests/test_role_free.py']},
        data_sha256={str(p):sha(p) for p in input_files},
        initial_full_sha256=full,initial_backbone_sha256=base,allocated_parameters=total_parameters,
        expected_sampling_sha256=sampling,positive_terms_per_epoch=positive_counts,
        fit_impressions=len(data.fit),valid_impressions=len(data.valid),test_impressions=len(data.test),
        all_models_from_scratch=True,test_used_for_selection=False,test_access_during_preflight=False,
        optional_not_queued=['rf_aux_only','nrms_ce'],
        historical_comparison_limitation='New unit is impression-mean CE; historical positive-example training is reference only',
        shared_budget='same seed, initialization, FIT impressions, all positives, negative samples, optimizer, five epochs; compute is measured, not equalized')
    save(ROOT/'FROZEN_PROTOCOL.json',protocol)
    save(logs/'COMPLETE.json',dict(time=now(),unit_tests=8,independent_history_checks=96,gpu_steps=4,
        real_fit_examples_only=True,test_labels_opened=False,protocol_sha256=sha(ROOT/'FROZEN_PROTOCOL.json')))
    print('PREFLIGHT_COMPLETE',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    try:main()
    except BaseException:
        import traceback
        save(ROOT/'PREFLIGHT_ERROR.json',dict(time=now(),traceback=traceback.format_exc()));raise
