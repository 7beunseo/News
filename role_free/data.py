"""Read-only reuse of frozen chronological inputs; no CBM outcome-memory input."""
from pathlib import Path
import numpy as np, torch, hashlib
from .config import ROOT,DATA_ROOT,CFG
from io_utils import read,save,sha,now

class Impressions:
    def __init__(self,dataset):
        assert dataset=='small', 'This protocol is MIND-small only'
        self.dataset=dataset
        self.prep=DATA_ROOT/dataset
        self.raw=self.prep
        self.info=read(self.prep/'COMPLETE.json')
        self.base_info=self.info
        self.a={n:np.load(self.raw/(n+'.npy'),mmap_mode='r') for n in ('items','offsets','users','timestamps','impressions','train_labels')}
        self.a['history']=np.load(self.prep/'matured.npy',mmap_mode='r')
        self.a['titles']=np.load(self.prep/'titles.npy',mmap_mode='r')
        total=len(self.a['history'])
        for n in ('users','timestamps','impressions'):self.a[n]=self.a[n][:total]
        self.a['offsets']=self.a['offsets'][:total+1]
        self.a['items']=self.a['items'][:int(self.a['offsets'][-1])]
        self.fit=np.load(self.prep/'train_origins.npy',mmap_mode='r')
        self.valid=np.load(self.prep/'valid_origins.npy',mmap_mode='r')
        d=self.base_info['splits']['dev'];self.test=np.arange(d['start'],d['start']+d['origins'],dtype=np.int64)
        self.lengths=self.a['offsets'][1:]-self.a['offsets'][:-1]

    def origins(self,split):return {'fit':self.fit,'valid':self.valid,'test':self.test}[split]

    def inputs(self,qq,device):
        qq=np.asarray(qq);ll=self.lengths[qq];width=max(1,int(ll.max(initial=0)))
        cc=np.zeros((len(qq),width),np.int64);mask=np.zeros_like(cc,bool)
        for j,q in enumerate(qq):
            lo,hi=map(int,self.a['offsets'][q:q+2]);cc[j,:hi-lo]=self.a['items'][lo:hi];mask[j,:hi-lo]=True
        h=torch.tensor(np.array(self.a['history'][qq],dtype=np.int64),device=device)
        return h,torch.tensor(cc,device=device),torch.tensor(mask,device=device)

    def fit_labels(self,qq,width,device):
        # VALID may call this helper for diagnostics only; training passes FIT.
        y=np.zeros((len(qq),width),np.float32)
        assert np.all(np.asarray(qq)<self.base_info['splits']['train']['origins'])
        for j,q in enumerate(qq):
            lo,hi=map(int,self.a['offsets'][q:q+2]);y[j,:hi-lo]=self.a['train_labels'][lo:hi]
        return torch.tensor(y,device=device)

    def microbatches(self,qq):
        st=0
        while st<len(qq):
            stop=st
            while stop<len(qq) and stop-st<CFG['microbatch_max']:
                nxt=stop+1;maxn=int(self.lengths[np.asarray(qq[st:nxt])].max())
                if stop>st and (nxt-st)*maxn*maxn>CFG['max_micro_pair_cells']:break
                stop=nxt
            yield qq[st:stop];st=stop

    def evaluation_labels(self,split,freeze_path=None):
        if split=='valid':return self.a['train_labels'],0
        assert split=='test' and freeze_path is not None and Path(freeze_path).exists()
        freeze=read(freeze_path);assert freeze['test_used_for_selection'] is False
        d=self.base_info['splits']['dev']
        out=ROOT/'data_audit'/self.dataset/'test_labels'
        if not (out/'COMPLETE.json').exists():
            import array
            source=Path(self.info['dev_behaviors'])
            assert sha(source)==self.info['dev_behaviors_sha256']
            lookup={n:i+1 for i,n in enumerate(read(self.raw/'NEWS_IDS.json'))};labels=array.array('b');count=0
            with source.open(encoding='utf-8') as f:
                for j,line in enumerate(f):
                    x=line.rstrip('\r\n').split('\t');q=int(self.test[j]);values=x[4].split()
                    assert int(x[0])==int(self.a['impressions'][q]);lo,hi=map(int,self.a['offsets'][q:q+2])
                    assert [lookup[v.rsplit('-',1)[0]] for v in values]==self.a['items'][lo:hi].tolist()
                    yy=[int(v.rsplit('-',1)[1]) for v in values];assert set(yy)<={0,1};labels.extend(yy);count+=1
            assert count==len(self.test) and len(labels)==d['candidates']
            out.mkdir(parents=True,exist_ok=True);np.save(out/'labels.npy',np.asarray(labels))
            save(out/'COMPLETE.json',dict(time=now(),origins=count,source_sha256=sha(source),freeze=str(freeze_path)))
        return np.load(out/'labels.npy',mmap_mode='r'),d['candidate_start']

    def audit(self):
        assert self.a['timestamps'][self.fit].max()<self.a['timestamps'][self.valid].min()
        assert self.a['timestamps'][self.valid].max()<self.a['timestamps'][self.test].min()
        scopes={}
        for name in ('fit','valid','test'):
            qq=self.origins(name);ll=self.lengths[qq]
            scopes[name]=dict(origins=len(qq),candidates=int(ll.sum()),max_candidates=int(ll.max()),
                candidate_quantiles=np.quantile(ll,[0,.5,.9,.99,1]).tolist(),origin_sha256=hashlib.sha256(qq.tobytes()).hexdigest())
        # Independently reconstruct selected histories using only past FIT outcomes.
        a=self.a;provided=np.load(self.raw/'provided.npy',mmap_mode='r');fitmask=np.zeros(len(a['users']),bool);fitmask[self.fit]=True
        order=np.lexsort((a['impressions'],a['timestamps'],a['users']));sorted_users=a['users'][order]
        checks=np.random.default_rng(26111877).choice(np.r_[self.fit,self.valid,self.test],96,replace=False)
        for q in checks:
            lo=np.searchsorted(sorted_users,a['users'][q],side='left');hi=np.searchsorted(sorted_users,a['users'][q],side='right')
            rr=order[lo:hi];past=rr[fitmask[rr]&(a['timestamps'][rr]<a['timestamps'][q]-86400)]
            hist=[int(i) for i in provided[q] if i>0]
            for s in past:
                l,r=map(int,a['offsets'][s:s+2]);hist.extend(map(int,a['items'][l:r][a['train_labels'][l:r]>0]))
            assert hist[-50:]==a['history'][q][a['history'][q]>0].tolist()
        result=dict(time=now(),dataset=self.dataset,scopes=scopes,vocab=self.info['vocab_size'],
            independent_history_checks=len(checks),test_labels_opened=False,history_definition='same frozen NRMS matured history; FIT-only extra clicks, strictly more than 24h old',
            metadata_assumption='Provided title/history available before impression; within-history click timestamps unavailable',
            hashes={str(self.prep/n):sha(self.prep/n) for n in ('titles.npy','matured.npy','train_origins.npy','valid_origins.npy')})
        save(ROOT/'data_audit'/f'{self.dataset}.json',result);return result
