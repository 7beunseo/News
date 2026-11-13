"""Causal data audit, CPU checks, real FIT GPU smoke and immutable protocol."""
import gc,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .config import ROOT,CODE,REPO,PREPARED,CFG,SEEDS,RUN_ORDER
from .experiment import CEData,seed_all,new_model,state_hash,candidates_for,step
from io_utils import read,save,sha,now


def main():
    if (ROOT/'FROZEN_PROTOCOL.json').exists():raise RuntimeError('Existing protocol preserved')
    ROOT.mkdir(parents=True,exist_ok=True);pre=ROOT/'preflight';pre.mkdir(exist_ok=True)
    check=subprocess.run([sys.executable,'-u','-B','-X','utf8','-m','unittest','discover','-s',str(REPO/'tests'),
        '-p','test_hccn.py','-v'],cwd=REPO,text=True,encoding='utf-8',stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (pre/'UNIT_TESTS.txt').write_text(check.stdout,encoding='utf-8');print(check.stdout,flush=True)
    if check.returncode:raise RuntimeError('CPU tests failed')
    data=CEData();audit=data.audit();n=len(data.a['positive_items'])
    hashes={};initial={};parameter_counts={}
    for seed in SEEDS:
        initial[str(seed)]={};hashes[str(seed)]={}
        for mode in RUN_ORDER:
            seed_all(seed);net=new_model(data.info['vocab_size'],mode);digest=state_hash(net)
            family='nrms_ce' if mode=='nrms_ce' else 'hccn_full'
            if family in initial[str(seed)]:assert initial[str(seed)][family]==digest
            initial[str(seed)][family]=digest;parameter_counts[mode]=sum(p.numel() for p in net.parameters())
            del net
        for epoch in range(1,6):
            rng=np.random.default_rng(seed+epoch);order=rng.permutation(n);digest=hashlib.sha256()
            for st in range(0,n,CFG['batch_size']):
                qq,cc=candidates_for(data,order[st:st+CFG['batch_size']],rng)
                digest.update(np.asarray(qq,dtype=np.int64).tobytes());digest.update(cc.tobytes())
            hashes[str(seed)][str(epoch)]=digest.hexdigest()
        print(json.dumps(dict(seed=seed,positive_examples=n,sampling_sha256=hashes[str(seed)])),flush=True)
    assert torch.cuda.is_available();titles=torch.tensor(np.array(data.a['titles'],dtype=np.int64),device='cuda')
    largest_origin=int(data.fit[np.argmax(data.lengths[data.fit])])
    largest_index=int(np.flatnonzero(data.a['positive_origins']==largest_origin)[0])
    typical=np.flatnonzero((data.lengths[data.a['positive_origins']]>=30)&(data.lengths[data.a['positive_origins']]<=50))[:2]
    gpu=[]
    for mode in RUN_ORDER:
        seed_all(SEEDS[0]);net=new_model(data.info['vocab_size'],mode).cuda().train();opt=torch.optim.Adam(net.parameters(),lr=CFG['lr'])
        chosen=np.array([largest_index]) if mode=='hccn_full' else typical
        qq,cc=candidates_for(data,chosen,np.random.default_rng(43));results=[]
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(3):results.append(step(net,data,titles,qq,cc,opt))
        torch.cuda.synchronize()
        gradients={}
        if mode=='hccn_full':
            for name,module in [('news',net.news),('memory',net.memory),('pair_query',net.graph.pair_query),
                ('rho',net.graph.rho),('competition',net.graph.competition)]:
                gradients[name]=sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
            assert all(v>0 and np.isfinite(v) for v in gradients.values()),gradients
        net.eval();h,c,mask=data.inputs(qq,'cuda')
        with torch.inference_mode():
            raw=net(titles,h,c,mask)['scores']
            ids=torch.unique(torch.cat([h.flatten(),c.flatten()]));cache=torch.zeros(len(titles),256,device='cuda')
            present=ids[ids>0]
            for j in range(0,len(present),64):cache[present[j:j+64]]=net.encode_news(titles[present[j:j+64]])
            cached=net.cached(cache,h,c,mask)['scores'];parity=float((raw-cached).abs().max())
            assert torch.allclose(raw,cached,atol=4e-4,rtol=1e-4),parity
        entry=dict(mode=mode,steps=3,origins=qq.tolist(),candidate_counts=data.lengths[qq].tolist(),
            losses=results,seconds=time.perf_counter()-t,peak_cuda_MiB=torch.cuda.max_memory_allocated()/2**20,
            gradients=gradients,cached_raw_max_error=parity)
        gpu.append(entry);print(json.dumps(entry),flush=True)
        del net,opt,cache,raw,cached;gc.collect();torch.cuda.empty_cache()
    del titles;gc.collect();torch.cuda.empty_cache()
    source_files=[*sorted(CODE.glob('*.py')),REPO/'config.py',REPO/'nrms.py',REPO/'io_utils.py',REPO/'metrics.py',REPO/'tests/test_hccn.py']
    inputs=[PREPARED/(x+'.npy') for x in ('items','offsets','users','timestamps','impressions','train_labels','matured',
        'titles','provided','train_origins','valid_origins')]+[PREPARED/'NEWS_IDS.json',PREPARED/'COMPLETE.json']
    protocol=dict(time=now(),config=CFG,seeds=list(SEEDS),run_order=list(RUN_ORDER),dataset='MIND-small',
        prepared_directory=str(PREPARED),source_sha256={str(p):sha(p) for p in source_files},data_sha256={str(p):sha(p) for p in inputs},
        initial_full_sha256=initial,expected_sampling_sha256=hashes,parameters=parameter_counts,
        fit_impressions=len(data.fit),positive_examples=n,valid_impressions=len(data.valid),test_impressions=len(data.test),
        planned_fits=len(SEEDS)*len(RUN_ORDER),previous_weights_loaded=False,test_used_for_selection=False,
        evaluation='All six selections frozen within each seed before reused research TEST; all seeds/architectures fixed before any TEST',
        assumptions=['Word positional embeddings learned; feedforward dimension 512; dropout 0.2; Adam 0.001',
            'A3 uses arithmetic mean of interest slots; A5 zeros scalar rho only, leaving relation-informed competition weights',
            'A0 reimplementation, random word init; new A0 full-context encoding protocol differs from historical sampled-only computation',
            'No-Rho ablation does not remove all nonseparable effects; unequal active parameters and FLOPs'])
    save(pre/'GPU_SMOKE.json',dict(time=now(),gpu=torch.cuda.get_device_name(),measurements=gpu,discarded_weights=True))
    save(ROOT/'FROZEN_PROTOCOL.json',protocol)
    save(pre/'COMPLETE.json',dict(time=now(),unit_tests=9,history_checks=audit['independent_history_checks'],
        gpu_steps=18,test_labels_opened=False,independent_training_fits=0))
    print('PREFLIGHT_COMPLETE',flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    try:main()
    except BaseException:
        import traceback
        save(ROOT/'PREFLIGHT_ERROR.json',dict(time=now(),traceback=traceback.format_exc()));raise
