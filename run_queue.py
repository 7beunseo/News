"""One dataset at a time, four controlled conditions, one seed, five epochs."""
import argparse
import subprocess
import sys
import numpy as np
from config import ROOT,CFG,SEEDS,RUN_ORDER
from io_utils import now,read,save
from train import verify_protocol


def execute(dataset):
    assert 1 <= CFG['epochs'] <= 5, 'Comparisons use at most five epochs.'
    verify_protocol();assert dataset in read(ROOT/'FROZEN_PROTOCOL.json')['datasets']
    for seed in SEEDS:
        selected={};orders=None;initial=None
        for mode in RUN_ORDER:
            run=ROOT/'runs'/dataset/str(seed)/mode
            command=[sys.executable,'-u','-B',str(ROOT/'train.py'),'--dataset',dataset,'--mode',mode,'--seed',str(seed)]
            if not (run/'TRAINING_COMPLETE.json').exists():subprocess.run(command,check=True,cwd=ROOT)
            row=read(run/'TRAINING_COMPLETE.json');selected[mode]=row
            completed=read(run/'EPOCHS.json')
            assert [x['epoch'] for x in completed] == list(range(1,CFG['epochs']+1))
            assert 1 <= row['selected_epoch'] <= CFG['epochs']
            if orders is None:orders=row['origin_order_hashes'];initial=row['backbone_initial_sha256']
            assert row['origin_order_hashes']==orders and row['backbone_initial_sha256']==initial
        save(ROOT/'frozen'/dataset/f'seed_{seed}.json',dict(time=now(),models=selected,
            validation_selected_model=max(RUN_ORDER,key=lambda m:selected[m]['validation_ndcg10']),test_used_for_selection=False))
        for mode in RUN_ORDER:
            if not (ROOT/'runs'/dataset/str(seed)/mode/'research_TEST/COMPLETE.json').exists():
                subprocess.run([sys.executable,'-u','-B',str(ROOT/'train.py'),'--dataset',dataset,'--mode',mode,'--seed',str(seed),'--test'],check=True,cwd=ROOT)
    values={m:float(np.mean([read(ROOT/'runs'/dataset/str(s)/m/'TRAINING_COMPLETE.json')['validation_ndcg10'] for s in SEEDS])) for m in RUN_ORDER}
    save(ROOT/'frozen'/dataset/'FINAL_SELECTION.json',dict(time=now(),winner=max(RUN_ORDER,key=lambda m:values[m]),
        mean_validation_ndcg10=values,test_used_for_selection=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['small','large'],required=True);a=p.parse_args();execute(a.dataset)
