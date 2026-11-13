"""Role-free, edge-supervised set reranking. Current outcomes never enter forward."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from nrms import NRMS


class AntisymmetricRelation(nn.Module):
    """Exact algebraic factorization of F([a,b,a*b,|a-b|,u,c]) - F(swapped).

    This avoids materializing the six-part concatenation for every pair; it is
    not a low-rank approximation or a new mathematical ranking objective.
    """
    def __init__(self,dim):
        super().__init__();self.dim=dim
        self.first=nn.Linear(6*dim,2*dim)
        self.last=nn.Linear(2*dim,dim,bias=False)

    def prepare(self,x,u,c):
        w=self.first.weight;d=self.dim
        return (F.linear(x,w[:,:d]),F.linear(x,w[:,d:2*d]),
                F.linear(u,w[:,4*d:5*d])+F.linear(c,w[:,5*d:])+self.first.bias)

    def block(self,x,u,c,lo,hi,prepared=None):
        d=self.dim;w=self.first.weight
        a,b,context=self.prepare(x,u,c) if prepared is None else prepared
        left=x[:,lo:hi,None,:];right=x[:,None,:,:]
        shared=F.linear(left*right,w[:,2*d:3*d])+F.linear((left-right).abs(),w[:,3*d:4*d])
        shared=shared+context[:,None,None,:]
        forward=F.gelu(a[:,lo:hi,None,:]+b[:,None,:,:]+shared)
        reverse=F.gelu(a[:,None,:,:]+b[:,lo:hi,None,:]+shared)
        return .5*self.last(forward-reverse)

    def dense_reference(self,x,u,c):
        b,n,d=x.shape;a=x[:,:,None,:].expand(b,n,n,d);z=x[:,None,:,:].expand_as(a)
        shared=[a*z,(a-z).abs(),u[:,None,None,:].expand_as(a),c[:,None,None,:].expand_as(a)]
        return .5*(self.last(F.gelu(self.first(torch.cat([a,z,*shared],-1))))-
                   self.last(F.gelu(self.first(torch.cat([z,a,*shared],-1)))))


class RelationAttention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();assert dim%heads==0
        self.dim=dim;self.heads=heads;self.head_dim=dim//heads
        self.q=nn.Linear(dim,dim,bias=False);self.k=nn.Linear(dim,dim,bias=False)
        self.v=nn.Linear(dim,dim,bias=False);self.edge_bias=nn.Linear(dim,heads,bias=False)
        self.edge_value=nn.Linear(dim,dim,bias=False);self.out=nn.Linear(dim,dim,bias=False)
        self.update=nn.Sequential(nn.Linear(3*dim,2*dim),nn.GELU(),nn.Linear(2*dim,dim))
        self.norm=nn.LayerNorm(dim)

    def block(self,x,mask,edges,lo,hi,keys,values):
        b,n,d=x.shape;h=self.heads;k=self.head_dim
        q=self.q(x[:,lo:hi]).reshape(b,hi-lo,h,k).transpose(1,2)
        logits=torch.einsum('bhqd,bhnd->bhqn',q,keys).float()/math.sqrt(k)
        if edges is not None:logits=logits+self.edge_bias(edges).permute(0,3,1,2).float()
        keep=mask[:,lo:hi,None]&mask[:,None,:]
        keep=keep&(torch.arange(lo,hi,device=x.device)[:,None]!=torch.arange(n,device=x.device)[None,:])
        logits=logits.masked_fill(~keep[:,None],-torch.inf)
        # Null neighbor: zero value and zero logit; prevents singleton/empty NaNs.
        weights=torch.cat([logits,logits.new_zeros(b,h,hi-lo,1)],-1).softmax(-1)[...,:n]
        message=torch.einsum('bhqn,bhnd->bhqd',weights.to(values.dtype),values)
        if edges is not None:
            value=self.edge_value(edges).reshape(b,hi-lo,n,h,k).permute(0,3,1,2,4)
            message=message+(weights.to(value.dtype)[...,None]*value).sum(-2)
        message=self.out(message.transpose(1,2).reshape(b,hi-lo,d))
        old=x[:,lo:hi];mass=weights.sum(-1).mean(1)[...,None]
        delta=self.update(torch.cat([old,message,old*message],-1)).tanh()
        updated=self.norm(old+mass.to(old.dtype)*delta)
        return torch.where(keep.any(-1)[...,None],updated,old)*mask[:,lo:hi,None]


class RoleFreeRanker(NRMS):
    def __init__(self,vocab_size,mode,relation_dim=64,heads=4,rounds=2,pair_chunk=32,news_chunk=256):
        super().__init__(vocab_size)
        self.mode=mode;self.rounds=rounds;self.pair_chunk=pair_chunk;self.news_chunk=news_chunk
        r=relation_dim;self.r=r
        self.news_project=nn.Sequential(nn.Linear(256,r),nn.LayerNorm(r))
        self.user_project=nn.Sequential(nn.Linear(256,r),nn.LayerNorm(r))
        self.initial_norm=nn.LayerNorm(r)
        self.relation=AntisymmetricRelation(r)
        self.edge_score=nn.Linear(r,1,bias=False)
        self.attention=RelationAttention(r,heads)
        self.head=nn.Sequential(nn.Linear(4*r,2*r),nn.GELU(),nn.Linear(2*r,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.beta_raw=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))

    def from_vectors(self,history,history_mask,candidates,mask):
        user=self.encode_user(history,history_mask)
        base=(candidates*user[:,None,:]).sum(-1).float().masked_fill(~mask,0)
        b,n,_=candidates.shape
        if self.mode in ('nrms_ce','nrms_ce_pair') or n==0:
            return dict(scores=base,base_logits=base,mask=mask,relation_delta=base*0,
                        edge_logits=base.new_zeros(b,n,n))
        x=self.news_project(candidates)*mask[...,None];u=self.user_project(user)
        initial=self.initial_norm(x+u[:,None,:])*mask[...,None];state=initial
        use_relations=self.mode!='plain_attention_pair'
        use_attention=self.mode!='rf_aux_only'
        iterations=self.rounds if use_attention else 1
        edge_logits=None
        for round_index in range(iterations):
            context=state.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            prepared=self.relation.prepare(state,u,context) if use_relations else None
            h=self.attention.heads;d=self.attention.head_dim
            keys=self.attention.k(state).reshape(b,n,h,d).transpose(1,2) if use_attention else None
            values=self.attention.v(state).reshape(b,n,h,d).transpose(1,2) if use_attention else None
            updated=[];rhos=[]
            for lo in range(0,n,self.pair_chunk):
                hi=min(n,lo+self.pair_chunk)
                keep=mask[:,lo:hi,None]&mask[:,None,:]
                keep=keep&(torch.arange(lo,hi,device=mask.device)[:,None]!=torch.arange(n,device=mask.device)[None,:])
                edges=self.relation.block(state,u,context,lo,hi,prepared)*keep[...,None] if use_relations else None
                if round_index==iterations-1:
                    rhos.append(self.edge_score(edges).squeeze(-1).float() if edges is not None else base.new_zeros(b,hi-lo,n))
                if use_attention:updated.append(self.attention.block(state,mask,edges,lo,hi,keys,values))
            if use_attention:state=torch.cat(updated,1)
            if rhos:edge_logits=torch.cat(rhos,1)
        delta=state-initial;ub=u[:,None,:].expand_as(x)
        value=self.head(torch.cat([x,ub,delta,delta*x],-1)).squeeze(-1).float()
        anchor=self.head(torch.cat([x,ub,torch.zeros_like(delta),torch.zeros_like(delta)],-1)).squeeze(-1).float()
        correction=F.softplus(self.beta_raw)*(value-anchor)*mask*(mask.sum(1)>1)[:,None]
        return dict(scores=(base+correction).masked_fill(~mask,0),base_logits=base,
                    mask=mask,edge_logits=edge_logits,relation_delta=correction)

    def forward(self,titles,history_ids,candidate_ids,candidate_mask):
        b,h=history_ids.shape;n=candidate_ids.shape[1]
        ids=torch.cat([history_ids,candidate_ids.masked_fill(~candidate_mask,0)],1).flatten()
        keep=ids.ne(0)
        if keep.any():
            tokens=titles[ids[keep]]
            encoded=torch.cat([self.encode_news(tokens[i:i+self.news_chunk]) for i in range(0,len(tokens),self.news_chunk)],0)
            vectors=encoded.new_zeros(len(ids),256);vectors[keep]=encoded
        else:vectors=self.words.weight.new_zeros(len(ids),256)
        vectors=vectors.reshape(b,h+n,256)
        return self.from_vectors(vectors[:,:h],history_ids.ne(0),vectors[:,h:],candidate_mask)

    def cached(self,cache,history_ids,candidate_ids,candidate_mask):
        return self.from_vectors(cache[history_ids],history_ids.ne(0),
            cache[candidate_ids.masked_fill(~candidate_mask,0)],candidate_mask)


def training_loss(output,labels,owners,columns,mode,relation_weight=.2,pair_weight=.2):
    scores=output['scores'].float();mask=output['mask'];b,n=scores.shape
    y=torch.where(mask,labels,torch.zeros_like(labels))
    counts=torch.bincount(owners,minlength=b).to(scores.dtype)
    picked=scores[owners[:,None],columns]
    ce_rows=F.cross_entropy(picked,torch.zeros(len(owners),device=scores.device,dtype=torch.long),reduction='none')
    ce=torch.zeros(b,device=scores.device).scatter_add(0,owners,ce_rows)/counts.clamp_min(1)
    pn=(mask&(y>0))[:,:,None]&(mask&(y==0))[:,None,:]
    denominator=pn.sum((1,2)).clamp_min(1)
    relation=(F.softplus(-output['edge_logits'].float())*pn).sum((1,2))/denominator
    pair=(F.softplus(-(scores[:,:,None]-scores[:,None,:]))*pn).sum((1,2))/denominator
    wr=relation_weight if mode in ('rf_full','rf_aux_only') else 0.
    wp=pair_weight if mode in ('nrms_ce_pair','plain_attention_pair') else 0.
    active=counts>0
    total=((ce+wr*relation+wp*pair)*active).sum()/active.sum().clamp_min(1)
    return total,dict(ce=ce.mean(),relation=relation.mean(),pair=pair.mean())


def potential_residual_ratio(rho,mask):
    """L2 residual to rho_ij = v_i-v_j. Diagnostic only, not a training loss.

    Zero residual means a scalar-potential difference on this candidate set.
    A nonzero residual alone is NOT evidence of useful recommendation signal.
    """
    keep=mask[:,:,None]&mask[:,None,:]
    r=rho.float()*keep
    potential=r.sum(-1)/mask.sum(1,keepdim=True).clamp_min(1)
    difference=potential[:,:,None]-potential[:,None,:]
    numerator=((r-difference).square()*keep).sum((1,2))
    denominator=r.square().sum((1,2))
    return numerator/denominator.clamp_min(1e-12),denominator
