"""Validate prepared inputs and freeze source/config before starting new fits."""
import argparse
from config import ROOT, CFG, SEEDS, RUN_ORDER
from data import Impressions
from io_utils import now,save,sha

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',choices=['small','large'],required=True);a=p.parse_args()
    assert 1 <= CFG['epochs'] <= 5,'Comparisons use at most five epochs.'
    assert not (ROOT/'FROZEN_PROTOCOL.json').exists(),'Existing frozen protocol preserved. Use a new checkout for a new protocol.'
    for dataset in a.datasets:Impressions(dataset).audit()
    sources=[p for p in ROOT.glob('*.py')]+list((ROOT/'tests').glob('*.py'))
    save(ROOT/'FROZEN_PROTOCOL.json',dict(time=now(),config=CFG,seeds=SEEDS,modes=RUN_ORDER,datasets=a.datasets,
        source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources},test_used_for_selection=False,
        split='chronological official TRAIN ~90/10 FIT/VALID; full official DEV research TEST'))
    print('Protocol frozen. No TEST labels opened.')
