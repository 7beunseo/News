"""Evaluate the completed VALID-best snapshot; an explicit --interim permits an unfinished fit."""
import argparse
from pathlib import Path
import numpy as np
import torch
from config import ROOT,CFG
from data import Impressions
from io_utils import now,read,save,sha
from train import verify_protocol,seed_all,new_model,evaluate,run_path
from model import MODES
from selection import select_validation_best

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['small','large'],required=True)
    p.add_argument('--mode',choices=MODES,default='rcrin_full');p.add_argument('--seed',type=int,default=26111801)
    p.add_argument('--interim',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    verify_protocol();run=run_path(a.dataset,a.seed,a.mode)
    if not a.interim:assert (run/'TRAINING_COMPLETE.json').exists(),'Fit unfinished; only explicit --interim permits early TEST.'
    records=read(run/'EPOCHS.json');best=select_validation_best(records,CFG['epochs'])
    assert sha(best['checkpoint'])==best['checkpoint_sha256']
    out=a.out.resolve();assert out.is_relative_to(ROOT/'manual_evaluations')
    out.mkdir(parents=True,exist_ok=False);freeze=out/'FROZEN_SELECTION.json'
    save(freeze,dict(time=now(),selected=best,test_used_for_selection=False,interim=a.interim,
        completed_epochs=[r['epoch'] for r in records],epoch_budget=CFG['epochs'],
        eligible_epochs=[r['epoch'] for r in records if 1<=r['epoch']<=CFG['epochs']],
        official_test=False,independent_unseen_test=False))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;seed_all(a.seed)
    data=Impressions(a.dataset);net=new_model(data.info['vocab_size'],a.mode).cuda()
    state=torch.load(best['checkpoint'],map_location='cpu',weights_only=False)
    assert state['epoch']==best['epoch'] and state['mode']==a.mode and state['seed']==a.seed
    net.load_state_dict(state['model']);del state;net.requires_grad_(False)
    titles=torch.tensor(np.array(data.a['titles'],dtype=np.int64),device='cuda')
    result=evaluate(net,data,titles,'test',out/'research_TEST',out,freeze)
    assert sha(best['checkpoint'])==best['checkpoint_sha256']
    save(out/'COMPLETE.json',dict(time=now(),selected_epoch=best['epoch'],result=result));print(result)
