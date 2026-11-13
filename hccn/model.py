"""Independent title-to-choice model. No NRMS score or current outcome input."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(logits,mask,dim=-1):
    p=logits.float().masked_fill(~mask,-1e9).softmax(dim)*mask
    return p/p.sum(dim,keepdim=True).clamp_min(1e-12)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer, without candidate-order positional features."""
    def __init__(self,dim,heads,feedforward,dropout):
        super().__init__();self.heads=heads;self.dk=dim//heads;assert dim%heads==0
        self.norm1=nn.LayerNorm(dim);self.qkv=nn.Linear(dim,3*dim);self.out=nn.Linear(dim,dim)
        self.norm2=nn.LayerNorm(dim);self.ff=nn.Sequential(nn.Linear(dim,feedforward),nn.GELU(),
            nn.Dropout(dropout),nn.Linear(feedforward,dim))
        self.drop=nn.Dropout(dropout)

    def forward(self,x,mask):
        b,n,d=x.shape
        if not n:return x
        q,k,v=self.qkv(self.norm1(x)).reshape(b,n,3,self.heads,self.dk).permute(2,0,3,1,4).unbind(0)
        safe=mask.clone();safe[:,0]|=~safe.any(1)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=safe[:,None,None,:],
            dropout_p=self.drop.p if self.training else 0.)
        x=x+self.drop(self.out(a.transpose(1,2).reshape(b,n,d)))
        return (x+self.drop(self.ff(self.norm2(x))))*mask[...,None]


class NewsTransformer(nn.Module):
    def __init__(self,vocab,word_dim=300,dim=256,layers=2,heads=4,feedforward=512,dropout=.2,max_tokens=32):
        super().__init__();self.dim=dim;self.max_tokens=max_tokens
        self.words=nn.Embedding(vocab,word_dim,padding_idx=0)
        nn.init.normal_(self.words.weight,std=.05)
        with torch.no_grad():self.words.weight[0].zero_()
        self.project=nn.Linear(word_dim,dim);self.position=nn.Embedding(max_tokens,dim)
        nn.init.normal_(self.position.weight,std=.02)
        self.drop=nn.Dropout(dropout)
        self.layers=nn.ModuleList([TransformerBlock(dim,heads,feedforward,dropout) for _ in range(layers)])
        self.norm=nn.LayerNorm(dim);self.pool=nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,1,bias=False))

    def forward(self,titles):
        b,n=titles.shape;assert n<=self.max_tokens
        if not n:return self.words.weight.new_zeros(b,self.dim)
        mask=titles.ne(0)
        x=self.drop(self.project(self.words(titles))+self.position(torch.arange(n,device=titles.device)))*mask[...,None]
        for layer in self.layers:x=layer(x,mask)
        x=self.norm(x)*mask[...,None];w=masked_softmax(self.pool(x).squeeze(-1),mask)
        return (x*w.to(x.dtype)[...,None]).sum(1)


class InterestMemory(nn.Module):
    def __init__(self,dim,slots):
        super().__init__();self.queries=nn.Parameter(torch.randn(slots,dim)*.02)
        self.key=nn.Linear(dim,dim,bias=False);self.value=nn.Linear(dim,dim,bias=False)
        self.out=nn.Linear(dim,dim,bias=False);self.norm=nn.LayerNorm(dim)

    def forward(self,history,mask):
        b,h,d=history.shape
        if not h:return history.new_zeros(b,len(self.queries),d)
        logits=torch.einsum('kd,bhd->bkh',self.queries,self.key(history)).float()/math.sqrt(d)
        weights=masked_softmax(logits,mask[:,None,:])
        memory=self.out(torch.einsum('bkh,bhd->bkd',weights.to(history.dtype),self.value(history)))
        return self.norm(memory)*mask.any(1)[:,None,None]


class Evidence(nn.Module):
    """Internal, jointly learned candidate evidence; not a frozen base predictor."""
    def __init__(self,dim):
        super().__init__();self.query=nn.Linear(dim,64,bias=False);self.key=nn.Linear(dim,64,bias=False)
        self.value=nn.Linear(dim,dim,bias=False)
        self.score=nn.Sequential(nn.Linear(4*dim,128),nn.GELU(),nn.Linear(128,1))

    def forward(self,z,memory):
        logits=torch.einsum('bnd,bkd->bnk',self.query(z),self.key(memory)).float()/8.
        selected=torch.einsum('bnk,bkd->bnd',logits.softmax(-1).to(z.dtype),self.value(memory))
        return self.score(torch.cat([z,selected,z*selected,z-selected],-1)).squeeze(-1).float()


class PairGraph(nn.Module):
    """Symmetric interest selection, antisymmetric evidence, learned competition."""
    def __init__(self,dim,r,hidden):
        super().__init__();self.r=r
        self.project=nn.Sequential(nn.Linear(dim,r),nn.LayerNorm(r))
        self.interest_key=nn.Linear(dim,r,bias=False);self.interest_value=nn.Linear(dim,r,bias=False)
        self.pair_query=nn.Sequential(nn.Linear(2*r,r),nn.GELU(),nn.Linear(r,r))
        self.first=nn.Linear(6*r,hidden);self.last=nn.Linear(hidden,r,bias=False)
        self.rho=nn.Linear(r,1,bias=False)
        self.competition=nn.Sequential(nn.Linear(4*r,hidden),nn.GELU(),nn.Linear(hidden,1))
        # Exact neutral readout at initialization: uniform omega and rho=0.
        nn.init.zeros_(self.rho.weight)
        nn.init.zeros_(self.competition[-1].weight);nn.init.zeros_(self.competition[-1].bias)

    def prepare(self,z,memory,mask):
        x=self.project(z)*mask[...,None];context=x.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        w=self.first.weight;r=self.r
        return dict(x=x,key=self.interest_key(memory),value=self.interest_value(memory),context=context,
            left=F.linear(x,w[:,:r]),right=F.linear(x,w[:,r:2*r]),
            shared_context=F.linear(context,w[:,5*r:])+self.first.bias)

    def block(self,p,lo,hi,common=False):
        x=p['x'];b,n,r=x.shape
        a=x[:,lo:hi,None,:];z=x[:,None,:,:]
        selection=p.get('selection_x',x)
        qa=selection[:,lo:hi,None,:];qz=selection[:,None,:,:]
        total=qa+qz;difference=(a-z).abs()
        if common:
            alpha=x.new_full((b,hi-lo,n,p['value'].shape[1]),1/p['value'].shape[1])
            interest=p['value'].mean(1)[:,None,None,:].expand(b,hi-lo,n,r)
        else:
            query=self.pair_query(torch.cat([total,(qa-qz).abs()],-1))
            logits=torch.einsum('bqnr,bkr->bqnk',query,p['key']).float()/math.sqrt(r)
            alpha=logits.softmax(-1)
            interest=torch.einsum('bqnk,bkr->bqnr',alpha.to(p['value'].dtype),p['value'])
        w=self.first.weight
        shared=F.linear(a*z,w[:,2*r:3*r])+F.linear(difference,w[:,3*r:4*r])
        shared=shared+F.linear(interest,w[:,4*r:5*r])+p['shared_context'][:,None,None,:]
        forward=F.gelu(p['left'][:,lo:hi,None,:]+p['right'][:,None,:,:]+shared)
        reverse=F.gelu(p['left'][:,None,:,:]+p['right'][:,lo:hi,None,:]+shared)
        relation=.5*self.last(forward-reverse)
        weight_logits=self.competition(torch.cat([a.expand(-1,-1,n,-1),z.expand(-1,hi-lo,-1,-1),interest,relation],-1)).squeeze(-1).float()
        rho=self.rho(relation).squeeze(-1).float()
        return rho,weight_logits,alpha,interest,relation

    def dense_relation_reference(self,p,interest):
        x=p['x'];b,n,r=x.shape;a=x[:,:,None,:].expand(b,n,n,r);z=x[:,None,:,:].expand_as(a)
        context=p['context'][:,None,None,:].expand_as(a)
        shared=[a*z,(a-z).abs(),interest,context]
        return .5*(self.last(F.gelu(self.first(torch.cat([a,z,*shared],-1))))-
                   self.last(F.gelu(self.first(torch.cat([z,a,*shared],-1)))))


def choice_readout(evidence,rho,weight_logits,mask,lo,hi):
    n=evidence.shape[1]
    valid=mask[:,lo:hi,None]&mask[:,None,:]
    valid=valid&(torch.arange(lo,hi,device=mask.device)[:,None]!=torch.arange(n,device=mask.device)[None,:])
    weights=masked_softmax(weight_logits,valid)
    edges=evidence[:,lo:hi,None]-evidence[:,None,:]+rho
    count=mask.sum(1,keepdim=True)
    score=(count-1).clamp_min(0)/count.clamp_min(1)*(weights*edges).sum(-1)
    score=torch.where(count==1,evidence[:,lo:hi],score)*mask[:,lo:hi]
    return score,weights,valid


class HCCN(nn.Module):
    def __init__(self,vocab_size,mode='hccn_full',word_dim=300,dim=256,news_layers=2,heads=4,
                 feedforward=512,slots=4,relation_dim=64,relation_hidden=128,dropout=.2,
                 pair_chunk=16,news_chunk=64,title_tokens=32):
        super().__init__();self.mode=mode;self.dim=dim;self.pair_chunk=pair_chunk;self.news_chunk=news_chunk
        assert mode in ('hccn_full','common_interest','individual','self_attention','no_rho')
        self.news=NewsTransformer(vocab_size,word_dim,dim,news_layers,heads,feedforward,dropout,title_tokens)
        self.memory=InterestMemory(dim,slots);self.evidence=Evidence(dim)
        self.graph=PairGraph(dim,relation_dim,relation_hidden)
        self.candidate_attention=TransformerBlock(dim,heads,feedforward,dropout)

    def encode_news(self,titles):return self.news(titles)

    def from_vectors(self,history,history_mask,z,mask,diagnostics=False,intervention=None):
        assert intervention in (None,'common','permute_pair_context')
        if intervention is not None:assert not self.training, 'Intervention is inference-only'
        memory=self.memory(history,history_mask);b,n,_=z.shape
        if self.mode=='self_attention':z=self.candidate_attention(z,mask)
        evidence=self.evidence(z,memory)*mask
        if self.mode in ('individual','self_attention') or n==0:
            result=dict(scores=evidence,evidence=evidence,mask=mask)
            if diagnostics:result.update(rho=evidence.new_zeros(b,n,n),omega=evidence.new_zeros(b,n,n),memory=memory)
            return result
        p=self.graph.prepare(z,memory,mask);scores=[];rr=[];ww=[];aa=[]
        if intervention=='permute_pair_context':
            indices=torch.arange(n,device=z.device).expand(b,n).clone()
            for row in range(b):
                valid=torch.nonzero(mask[row],as_tuple=False).flatten();indices[row,valid]=valid.flip(0)
            p['selection_x']=p['x'].gather(1,indices[...,None].expand_as(p['x']))
        for lo in range(0,n,self.pair_chunk):
            hi=min(n,lo+self.pair_chunk)
            rho,logits,alpha,_,_=self.graph.block(p,lo,hi,common=self.mode=='common_interest' or intervention=='common')
            if self.mode=='no_rho':rho=rho*0
            value,weights,keep=choice_readout(evidence,rho,logits,mask,lo,hi);scores.append(value)
            if diagnostics:rr.append(rho*keep);ww.append(weights);aa.append(alpha*keep[...,None])
        result=dict(scores=torch.cat(scores,1),evidence=evidence,mask=mask)
        if diagnostics:result.update(rho=torch.cat(rr,1),omega=torch.cat(ww,1),pair_weights=torch.cat(aa,1),memory=memory)
        return result

    def forward(self,titles,history_ids,candidate_ids,candidate_mask,diagnostics=False,intervention=None):
        b,h=history_ids.shape;n=candidate_ids.shape[1]
        ids=torch.cat([history_ids,candidate_ids.masked_fill(~candidate_mask,0)],1).flatten();keep=ids.ne(0)
        if keep.any():
            tokens=titles[ids[keep]]
            encoded=torch.cat([self.encode_news(tokens[i:i+self.news_chunk]) for i in range(0,len(tokens),self.news_chunk)],0)
            vectors=encoded.new_zeros(len(ids),self.dim);vectors[keep]=encoded
        else:vectors=self.news.words.weight.new_zeros(len(ids),self.dim)
        vectors=vectors.reshape(b,h+n,self.dim)
        return self.from_vectors(vectors[:,:h],history_ids.ne(0),vectors[:,h:],candidate_mask,diagnostics,intervention)

    def cached(self,cache,history_ids,candidate_ids,candidate_mask,diagnostics=False,intervention=None):
        return self.from_vectors(cache[history_ids],history_ids.ne(0),
            cache[candidate_ids.masked_fill(~candidate_mask,0)],candidate_mask,diagnostics,intervention)
