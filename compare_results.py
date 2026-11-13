"""Compare saved TEST metrics after confirming identical evaluated impressions."""
import argparse
import csv
from itertools import zip_longest
from pathlib import Path
import numpy as np
from config import ROOT
from io_utils import now, read, save, sha

KEYS = ('auc', 'mrr', 'ndcg5', 'ndcg10')


def compare(left, right, left_name='left', right_name='right'):
    left, right = Path(left), Path(right)
    a, b = read(left/'COMPLETE.json'), read(right/'COMPLETE.json')
    assert a['split'] == b['split'] == 'test', 'Both inputs must be TEST evaluations.'
    for key in ('origins', 'eligible_origins', 'candidates'):
        assert a[key] == b[key], f'Mismatched {key}'
    sums = np.zeros((2, 4)); count = 0
    with (left/'SESSIONS.csv').open(encoding='utf-8-sig') as lf, (right/'SESSIONS.csv').open(encoding='utf-8-sig') as rf:
        for l, r in zip_longest(csv.DictReader(lf), csv.DictReader(rf)):
            assert l is not None and r is not None, 'Different session counts'
            for key in ('origin', 'impression', 'user'):
                assert l[key] == r[key], f'Session {count}: mismatched {key}'
            sums[0] += [float(l[k]) for k in KEYS]
            sums[1] += [float(r[k]) for k in KEYS]
            count += 1
    assert count == a['eligible_origins'] and count > 0
    for i, result in enumerate((a, b)):
        assert np.allclose(sums[i]/count, [result[k] for k in KEYS], rtol=0, atol=1e-10), 'Saved aggregate differs from session mean'
    return dict(time=now(), left_name=left_name, right_name=right_name,
        left=a, right=b, matched_origins=count,
        delta_left_minus_right={k: a[k]-b[k] for k in KEYS},
        relative_percent_left_minus_right={k: (100*(a[k]/b[k]-1) if b[k] else None) for k in KEYS},
        same_test_impressions=True, training_conditions_not_checked=True,
        source_sha256={str(p):sha(p) for d in (left,right) for p in (d/'COMPLETE.json',d/'SESSIONS.csv')},
        limitation='Matching evaluation units does not establish matching training conditions or independent unseen TEST status.')


def compare_many(inputs):
    """Use the first named result as reference; preserve input order, not TEST rank."""
    if len(inputs)<2:raise ValueError('At least two named results are required.')
    names=list(inputs);reference=names[0];comparisons={};results={}
    for name in names[1:]:
        record=compare(inputs[name],inputs[reference],name,reference)
        comparisons[name]=record
        results[reference]=record['right'];results[name]=record['left']
    return dict(time=now(),reference_model=reference,matched_origins=results[reference]['eligible_origins'],
        rows=[dict(model=name,**{k:results[name][k] for k in KEYS}) for name in names],
        comparisons_vs_reference=comparisons,selection_performed=False,
        limitation='Saved TEST comparison only. Training budgets and independent replication are not inferred from this table.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--left-name',default='left');p.add_argument('--right-name',default='right')
    p.add_argument('--additional',nargs=2,action='append',metavar=('NAME','RESULT_DIR'),default=[])
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();assert out.is_relative_to(ROOT/'comparisons')
    inputs={args.left_name:args.left,args.right_name:args.right}
    assert len(inputs)==2,'Model names must be unique.'
    for name,path in args.additional:
        assert name not in inputs,'Model names must be unique.'
        inputs[name]=Path(path)
    result=compare_many(inputs) if args.additional else compare(args.left,args.right,args.left_name,args.right_name)
    out.mkdir(parents=True,exist_ok=False);save(out/'COMPARISON.json',result)
    with (out/'COMPARISON.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['model',*KEYS]);writer.writeheader()
        if args.additional:writer.writerows(result['rows'])
        else:
            for label,d in [(args.left_name,result['left']),(args.right_name,result['right'])]:
                writer.writerow(dict(model=label,**{k:d[k] for k in KEYS}))
    print(result['rows'] if args.additional else result['delta_left_minus_right'])
