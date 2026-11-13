"""Exact tie-aware metrics used by the recorded runs, plus independent audit."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

def metric(y,s):
 order=np.argsort(-s,kind='stable');z=np.asarray(s)[order];g=np.asarray(y,dtype=np.float64)[order]
 n=len(g);p=float(g.sum())
 if not 0<p<n:return None
 assert np.isfinite(z).all()
 starts=np.r_[0,np.flatnonzero(z[1:]!=z[:-1])+1];ends=np.r_[starts[1:],n]
 positives=np.add.reduceat(g,starts);expected=np.repeat(positives/(ends-starts),ends-starts)
 ranks=(starts+1+ends)/2;discount=1/np.log2(np.arange(2,n+2))
 return dict(auc=float((np.dot(positives,n+1-ranks)-p*(p+1)/2)/(p*(n-p))),
 mrr=float(np.dot(expected,1/np.arange(1,n+1))/p),
 ndcg5=float(np.dot(expected[:5],discount[:5])/discount[:min(int(p),5)].sum()),
 ndcg10=float(np.dot(expected[:10],discount[:10])/discount[:min(int(p),10)].sum()))

def independent_metrics(y,s):
    y=np.asarray(y,float);s=np.asarray(s,float);result={'auc':float(roc_auc_score(y,s))}
    # Average discount within tied score groups, rather than reuse training metrics.
    ranks=rankdata(-s,method='min').astype(int);gains=np.zeros(len(s))
    reciprocal=np.zeros(len(s));discount=1/np.log2(np.arange(2,len(s)+2))
    for value in np.unique(s):
        ix=np.flatnonzero(s==value);start=int(ranks[ix[0]])-1;positions=np.arange(start,start+len(ix))
        reciprocal[ix]=np.mean(1/(positions+1))
        for k in [5,10]:
            contrib=discount[positions]*(positions<k)
            result.setdefault('dcg'+str(k),0.)
            result['dcg'+str(k)]+=float(y[ix].sum()*contrib.mean())
    result['mrr']=float(np.dot(y,reciprocal)/y.sum())
    for k in [5,10]:result['ndcg'+str(k)]=result.pop('dcg'+str(k))/discount[:min(k,int(y.sum()))].sum()
    return result
