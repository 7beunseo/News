"""Frozen-weight VALID diagnostics; not a substitute for retrained ablations."""
import numpy as np
import torch
from io_utils import save,now
from metrics import metric


@torch.inference_mode()
def diagnose(model,data,titles,run):
    path=run/'VALID_GRAPH_DIAGNOSTICS.json'
    if path.exists():return
    if model.mode=='nrms_ce':
        save(path,dict(time=now(),applicable=False,reason='A0 has no competition graph'));return
    model.eval();origins=data.valid[np.linspace(0,len(data.valid)-1,128,dtype=int)]
    rows=[]
    for micro in data.microbatches(origins):
        h,c,mask=data.inputs(micro,titles.device);labels=data.fit_labels(micro,c.shape[1],titles.device)
        original=model(titles,h,c,mask,diagnostics=True)
        variants={'normal':original}
        if model.mode in ('hccn_full','no_rho'):
            variants.update({name:model(titles,h,c,mask,intervention=name) for name in ('common','permute_pair_context')})
        for j,q in enumerate(micro):
            valid=mask[j];n=int(valid.sum());rho=original['rho'][j][valid][:,valid]
            potential=rho.sum(-1)/max(1,n);residual=rho-(potential[:,None]-potential[None,:])
            energy=float(rho.square().sum())
            r=dict(origin=int(q),candidates=n,rho_energy=energy,
                nonpotential_ratio=float(residual.square().sum()/energy) if energy>1e-10 else None,
                metrics={})
            yy=labels[j][valid].cpu().numpy()
            for name,value in variants.items():r['metrics'][name]=metric(yy,value['scores'][j][valid].float().cpu().numpy())
            rows.append(r)
    save(path,dict(time=now(),split='VALID',selection_use=False,origins=len(rows),rows=rows,
        warning='Fixed-weight interventions are not retraining. Non-potential energy alone is not evidence of useful relations.'))
