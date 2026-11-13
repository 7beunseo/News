"""Prepare complete labelled TRAIN/DEV without consuming DEV outcomes.

Portable adaptation of the historical MIND-small preparation. Full large is
supported but historical large used an additional catalogue containing official
TEST metadata; this release intentionally does not require that catalogue.
"""
import argparse
import array
import calendar
from collections import Counter, deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import re
import numpy as np
from config import DATA_ROOT, CFG
from io_utils import now, save, sha


@lru_cache(maxsize=200000)
def timestamp(value):
    return calendar.timegm(datetime.strptime(value, '%m/%d/%Y %I:%M:%S %p').timetuple())


def prepare(train, dev, out, dataset):
    train,dev,out=Path(train).resolve(),Path(dev).resolve(),Path(out).resolve()
    assert dataset in ('small','large')
    for folder in (train,dev):
        for name in ('news.tsv','behaviors.tsv'):assert (folder/name).is_file(),folder/name
    out.mkdir(parents=True,exist_ok=False)
    save(out/'STARTED.json',dict(time=now(),dataset=dataset))
    hashes={str(folder/name):sha(folder/name) for folder in (train,dev) for name in ('news.tsv','behaviors.tsv')}
    news={}
    for folder in (train,dev):
        with (folder/'news.tsv').open(encoding='utf-8') as f:
            for line in f:
                parts=line.rstrip('\r\n').split('\t');assert len(parts)>=4
                news.setdefault(parts[0],parts[3])
    names=sorted(news);lookup={name:i+1 for i,name in enumerate(names)}
    save(out/'NEWS_IDS.json',names)
    specs=dict(items='I',offsets='q',users='i',timestamps='q',impressions='q',provided='I',train_labels='b')
    buf={k:array.array(v) for k,v in specs.items()};buf['offsets'].append(0)
    users={};ranges={};total=0
    for name,folder in [('train',train),('dev',dev)]:
        start=total;candidate_start=len(buf['items']);seen=set()
        with (folder/'behaviors.tsv').open(encoding='utf-8') as f:
            for line in f:
                parts=line.rstrip('\r\n').split('\t');assert len(parts)==5
                impression,user,clock,history,raw=parts;impression=int(impression)
                assert impression not in seen;seen.add(impression)
                values=raw.split();ids=[lookup[v.rsplit('-',1)[0]] for v in values]
                assert ids and len(ids)==len(set(ids))
                h=[lookup[v] for v in history.split()][-CFG['history']:]
                buf['provided'].extend(h+[0]*(CFG['history']-len(h)))
                buf['items'].extend(ids);buf['offsets'].append(len(buf['items']))
                buf['users'].append(users.setdefault(user,len(users)))
                buf['timestamps'].append(timestamp(clock));buf['impressions'].append(impression)
                if name=='train':
                    y=[int(v.rsplit('-',1)[1]) for v in values];assert set(y)<={0,1}
                    buf['train_labels'].extend(y)
                # DEV labels are discarded as strings, never converted to outcomes.
                total+=1
                if total%50000==0:print(f'INDEX {name}: {total} impressions',flush=True)
        ranges[name]=dict(start=start,origins=total-start,candidate_start=candidate_start,candidates=len(buf['items'])-candidate_start)
    a={k:np.asarray(v) for k,v in buf.items()};del buf
    a['provided']=a['provided'].reshape(total,CFG['history'])
    n=ranges['train']['origins'];ts=a['timestamps'];us=a['users'];imp=a['impressions']
    order=np.lexsort((imp[:n],ts[:n]));cutoff=int(ts[order[int(.9*n)]])
    fit=np.zeros(total,bool);fit[:n]=ts[:n]<cutoff
    valid=np.flatnonzero(~fit[:n]);test=np.arange(n,total,dtype=np.int64)
    assert fit.any() and len(valid) and len(test)
    assert ts[fit].max()<ts[valid].min() and ts[valid].max()<ts[test].min()
    a.update(fit_mask=fit,train_origins=np.flatnonzero(fit),valid_origins=valid,test_origins=test)
    seen=np.zeros(len(names)+1,bool)
    for q in np.flatnonzero(fit):
        lo,hi=map(int,a['offsets'][q:q+2]);seen[a['items'][lo:hi]]=True;seen[a['provided'][q]]=True
    seen[0]=False;token=re.compile(r'\w+|[.,!?;|]');freq=Counter()
    for i in np.flatnonzero(seen):freq.update(token.findall(news[names[i-1]].lower()))
    vocab={'<pad>':0,'<unk>':1}
    for word in sorted(w for w,count in freq.items() if count>=3):vocab[word]=len(vocab)
    titles=np.zeros((len(names)+1,CFG['title_tokens']),np.int32)
    for i,name in enumerate(names,1):
        words=[vocab.get(w,1) for w in token.findall(news[name].lower())[:CFG['title_tokens']]]
        titles[i,:len(words)]=words
    a['titles']=titles;save(out/'VOCAB.json',vocab)
    matured=np.zeros((total,CFG['history']),np.uint32)
    order=np.lexsort((imp,ts,us));sorted_users=us[order]
    cuts=np.r_[0,np.flatnonzero(sorted_users[1:]!=sorted_users[:-1])+1,total]
    for g in range(len(cuts)-1):
        ix=order[cuts[g]:cuts[g+1]];eligible=ix[fit[ix]];cursor=0;past=deque(maxlen=CFG['history'])
        for q in ix:
            while cursor<len(eligible) and ts[eligible[cursor]]<ts[q]-86400:
                s=int(eligible[cursor]);lo,hi=map(int,a['offsets'][s:s+2])
                past.extend(map(int,a['items'][lo:hi][a['train_labels'][lo:hi]>0]));cursor+=1
            current=([int(i) for i in a['provided'][q] if i>0]+list(past))[-CFG['history']:]
            matured[q,:len(current)]=current
        if g%10000==0:print(f'HISTORY {g}/{len(cuts)-1} users',flush=True)
    a['matured']=matured
    checks=np.random.default_rng(26111740).choice(total,min(96,total),replace=False)
    for q in checks:
        rr=order[np.searchsorted(sorted_users,us[q],side='left'):np.searchsorted(sorted_users,us[q],side='right')]
        past=rr[fit[rr]&(ts[rr]<ts[q]-86400)]
        expected=[int(i) for i in a['provided'][q] if i>0]
        for s in past:
            lo,hi=map(int,a['offsets'][s:s+2]);expected.extend(map(int,a['items'][lo:hi][a['train_labels'][lo:hi]>0]))
        assert expected[-CFG['history']:]==matured[q][matured[q]>0].tolist()
    for key,value in a.items():np.save(out/(key+'.npy'),value,allow_pickle=False)
    assert all(sha(p)==h for p,h in hashes.items())
    save(out/'COMPLETE.json',dict(time=now(),dataset=dataset,splits=ranges,vocab_size=len(vocab),
        fit_origins=int(fit.sum()),valid_origins=len(valid),research_test_origins=len(test),
        cutoff_timestamp=cutoff,dev_labels_parsed=False,official_test_used=False,
        dev_behaviors=str(dev/'behaviors.tsv'),dev_behaviors_sha256=hashes[str(dev/'behaviors.tsv')],
        source_sha256=hashes,independent_causal_checks=len(checks),
        catalogue='TRAIN and DEV news metadata only; titles use TRAIN-first precedence',
        clock_policy='common raw MIND timestamp axis; absolute timezone unspecified'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['small','large'],required=True)
    p.add_argument('--train-dir',type=Path,required=True);p.add_argument('--dev-dir',type=Path,required=True)
    a=p.parse_args();prepare(a.train_dir,a.dev_dir,DATA_ROOT/a.dataset,a.dataset)
