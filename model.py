"""RCRIN: full-impression reciprocal role interaction, no labels in forward."""
from pathlib import Path
import importlib.util, math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

import nrms as backbone

MODES=('nrms_bce','nrms_bce_pair','rcrin_p_from_n','rcrin_n_from_p',
       'rcrin_no_interaction','rcrin_state_concat','rcrin_no_exchange','rcrin_unanchored','rcrin_full')


def mlp(inp,hidden,out):
    return nn.Sequential(nn.Linear(inp,hidden),nn.GELU(),nn.Linear(hidden,out))


class CrossRoleMessage(nn.Module):
    def __init__(self,dim=64,heads=4):
        super().__init__();assert dim%heads==0
        self.dim=dim;self.heads=heads;self.head_dim=dim//heads
        self.q=nn.Linear(dim,dim,bias=False);self.k=nn.Linear(dim,dim,bias=False)
        self.v=nn.Linear(dim,dim,bias=False);self.edge_bias=nn.Linear(dim,heads,bias=False)
        self.edge_value=nn.Linear(dim,dim,bias=False);self.out=nn.Linear(dim,dim,bias=False)
        self.update=mlp(dim*3,dim*2,dim);self.gate=mlp(dim*3,dim,dim);self.norm=nn.LayerNorm(dim)

    def forward(self,target,source,edges,partner_role,mask,query_chunk=32,diagnostics=False):
        b,n,d=target.shape
        if n==0:return target,target.new_zeros(b,0)
        keys=self.k(source).reshape(b,n,self.heads,self.head_dim).transpose(1,2)
        values=self.v(source).reshape(b,n,self.heads,self.head_dim).transpose(1,2)
        outputs=[];masses=[]
        for lo in range(0,n,max(1,query_chunk)):
            hi=min(n,lo+max(1,query_chunk));q=self.q(target[:,lo:hi]).reshape(b,hi-lo,self.heads,self.head_dim).transpose(1,2)
            e=edges[:,lo:hi]
            logits=torch.einsum('bhqd,bhkd->bhqk',q,keys).float()/math.sqrt(self.head_dim)
            logits=logits+self.edge_bias(e).permute(0,3,1,2).float()+partner_role.clamp(1e-6,1-1e-6).log()[:,None,None,:]
            keep=mask[:,lo:hi,None]&mask[:,None,:]
            keep=keep & (torch.arange(lo,hi,device=mask.device)[:,None]!=torch.arange(n,device=mask.device)[None,:])
            logits=logits.masked_fill(~keep[:,None],-torch.inf)
            # Fixed zero-logit, zero-value null, exactly as the specified formula.
            probs=torch.cat([logits,logits.new_zeros(b,self.heads,hi-lo,1)],-1).softmax(-1)[...,:n]
            real_mass=probs.sum(-1).mean(1)
            edge_value=self.edge_value(e).reshape(b,hi-lo,n,self.heads,self.head_dim).permute(0,3,1,2,4)
            message=torch.einsum('bhqk,bhkd->bhqd',probs.to(values.dtype),values)
            message=message+(probs.to(edge_value.dtype)[...,None]*edge_value).sum(-2)
            message=self.out(message.transpose(1,2).reshape(b,hi-lo,d))
            h=target[:,lo:hi];z=torch.cat([h,message,h*message],-1)
            delta=self.update(z).tanh()*self.gate(z).sigmoid()
            new=self.norm(h+real_mass[...,None].to(h.dtype)*delta)
            # LN(h+0) is not necessarily h: explicitly preserve zero-partner states.
            new=torch.where(keep.any(-1)[...,None],new,h)*mask[:,lo:hi,None]
            outputs.append(new);masses.append(real_mass*mask[:,lo:hi])
        return torch.cat(outputs,1),torch.cat(masses,1)


class ReciprocalCrossRoleInteraction(nn.Module):
    def __init__(self,input_dim=256,relation_dim=64,heads=4,rounds=2,variant='rcrin_full',
                 pair_chunk=32,query_chunk=32,pair_checkpoint=True):
        super().__init__();self.dim=relation_dim;self.rounds=rounds;self.variant=variant
        self.pair_chunk=pair_chunk;self.query_chunk=query_chunk;self.pair_checkpoint=pair_checkpoint
        r=relation_dim
        self.news_project=nn.Sequential(nn.Linear(input_dim,r),nn.LayerNorm(r))
        self.user_project=nn.Sequential(nn.Linear(input_dim,r),nn.LayerNorm(r))
        self.role=mlp(3*r,2*r,1);self.role_plus=nn.Parameter(torch.zeros(r));self.role_minus=nn.Parameter(torch.zeros(r))
        nn.init.normal_(self.role_plus,std=.02);nn.init.normal_(self.role_minus,std=.02)
        self.initial_plus=nn.LayerNorm(r);self.initial_minus=nn.LayerNorm(r)
        self.pair=mlp(6*r,2*r,r);self.edge_score=nn.Linear(r,1,bias=False)
        self.positive_reads_negative=CrossRoleMessage(r,heads)
        self.negative_reads_positive=CrossRoleMessage(r,heads)
        self.fusion_gate=mlp(6*r,2*r,r);self.interaction=nn.Linear(r,r,bias=False)
        self.concat_fusion=mlp(2*r,2*r,r) if variant=='rcrin_state_concat' else None
        self.score=mlp(4*r,2*r,1)
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)
        self.beta_raw=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))

    def pair_block(self,xi,xj,u,c):
        b,q,d=xi.shape;n=xj.shape[1]
        a=xi[:,:,None,:].expand(b,q,n,d);z=xj[:,None,:,:].expand(b,q,n,d)
        user=u[:,None,None,:].expand_as(a);ctx=c[:,None,None,:].expand_as(a)
        shared=[a*z,(a-z).abs(),user,ctx]
        return .5*(self.pair(torch.cat([a,z,*shared],-1))-self.pair(torch.cat([z,a,*shared],-1)))

    def edges(self,x,u,c,mask):
        n=x.shape[1]
        if n==0:return x.new_zeros(x.shape[0],0,0,self.dim)
        parts=[]
        for lo in range(0,n,max(1,self.pair_chunk)):
            xi=x[:,lo:lo+self.pair_chunk]
            if self.training and self.pair_checkpoint and torch.is_grad_enabled():
                e=checkpoint(self.pair_block,xi,x,u,c,use_reentrant=False)
            else:e=self.pair_block(xi,x,u,c)
            parts.append(e)
        edge=torch.cat(parts,1)
        keep=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,device=x.device,dtype=torch.bool)[None]
        return edge*keep[...,None]

    def forward(self,candidate_vectors,user_vectors,candidate_mask,base_logits,return_diagnostics=False):
        b,n,_=candidate_vectors.shape;mask=candidate_mask.bool()
        if n==0:
            z=base_logits.float();return {'scores':z,'role_logits':z,'edge_logits':z.new_zeros(b,0,0),
                'relation_delta':z,'delta_plus':z.new_zeros(b,0,self.dim),'delta_minus':z.new_zeros(b,0,self.dim),'mask':mask}
        x=self.news_project(candidate_vectors)*mask[...,None];u=self.user_project(user_vectors)
        context=x.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        ub=u[:,None,:].expand_as(x);cb=context[:,None,:].expand_as(x)
        role_logits=self.role(torch.cat([x,ub,cb],-1)).squeeze(-1).float()*mask
        pi=role_logits.sigmoid().clamp(1e-6,1-1e-6)
        initial_p=self.initial_plus(x+ub+self.role_plus)*mask[...,None]
        initial_n=self.initial_minus(x+ub+self.role_minus)*mask[...,None]
        hp,hn=initial_p,initial_n;edges=self.edges(x,u,context,mask)
        rho=self.edge_score(edges).squeeze(-1).float();masses_p=[];masses_n=[]
        for _ in range(self.rounds):
            oldp,oldn=hp,hn
            if self.variant not in ('rcrin_n_from_p','rcrin_no_exchange'):
                hp,mp=self.positive_reads_negative(oldp,oldn,edges,1-pi,mask,self.query_chunk)
            else:hp=oldp;mp=pi*0
            if self.variant not in ('rcrin_p_from_n','rcrin_no_exchange'):
                hn,mn=self.negative_reads_positive(oldn,oldp,-edges,pi,mask,self.query_chunk)
            else:hn=oldn;mn=pi*0
            masses_p.append(mp);masses_n.append(mn)
        dp=hp-initial_p;dn=hn-initial_n;interaction=dp*dn;contrast=dp-dn
        if self.variant=='rcrin_no_interaction':interaction=interaction*0
        gate=self.fusion_gate(torch.cat([x,ub,dp,dn,interaction,contrast],-1)).sigmoid()
        if self.variant=='rcrin_state_concat':
            fused=self.concat_fusion(torch.cat([hp,hn],-1));head_contrast=contrast*0
        else:
            fused=gate*dp+(1-gate)*dn+self.interaction(interaction);head_contrast=contrast
        correction=self.score(torch.cat([x,ub,fused,head_contrast],-1)).squeeze(-1).float()
        if self.variant!='rcrin_unanchored':
            # Approved narrow modification: prevent an x/u-only MLP shortcut.
            anchor=self.score(torch.cat([x,ub,torch.zeros_like(fused),torch.zeros_like(head_contrast)],-1)).squeeze(-1).float()
            correction=correction-anchor
        correction=F.softplus(self.beta_raw)*correction*mask*(mask.sum(1)>1)[:,None]
        output={'scores':(base_logits.float()+correction).masked_fill(~mask,0),
                'role_logits':role_logits,'edge_logits':rho,'relation_delta':correction,
                'delta_plus':dp,'delta_minus':dn,'mask':mask}
        if return_diagnostics:output.update(edges=edges,role_prob=pi,role_mass_plus=masses_p,role_mass_minus=masses_n,fused=fused)
        return output


class RCRINRanker(backbone.NRMS):
    def __init__(self,vocab_size,mode='rcrin_full',relation_dim=64,heads=4,rounds=2,
                 pair_chunk=32,query_chunk=32,news_chunk=128,pair_checkpoint=True):
        super().__init__(vocab_size);assert mode in MODES
        self.mode=mode;self.news_chunk=news_chunk
        self.relation=None if mode.startswith('nrms_') else ReciprocalCrossRoleInteraction(
            256,relation_dim,heads,rounds,mode,pair_chunk,query_chunk,pair_checkpoint)

    def from_vectors(self,history,history_mask,candidates,candidate_mask,return_diagnostics=False):
        user=self.encode_user(history,history_mask)
        base=(candidates*user[:,None,:]).sum(-1).float()
        if self.relation is None:return {'scores':base.masked_fill(~candidate_mask,0),'mask':candidate_mask,'base_logits':base}
        result=self.relation(candidates,user,candidate_mask,base,return_diagnostics)
        result['base_logits']=base;return result

    def forward(self,titles,history_ids,candidate_ids,candidate_mask,return_diagnostics=False):
        b,h=history_ids.shape;n=candidate_ids.shape[1]
        candidate_ids=candidate_ids.masked_fill(~candidate_mask,0)
        allids=torch.cat([history_ids,candidate_ids],1);flat=allids.flatten();present=flat.ne(0)
        if present.any():
            tok=titles[flat[present]]
            enc=torch.cat([self.encode_news(tok[i:i+self.news_chunk]) for i in range(0,len(tok),self.news_chunk)],0)
            vectors=enc.new_zeros(len(flat),256);vectors[present]=enc
        else:vectors=self.words.weight.new_zeros(len(flat),256)
        vectors=vectors.reshape(b,h+n,256)
        return self.from_vectors(vectors[:,:h],history_ids.ne(0),vectors[:,h:],candidate_mask,return_diagnostics)

    def cached(self,cache,history_ids,candidate_ids,candidate_mask,return_diagnostics=False):
        candidate_ids=candidate_ids.masked_fill(~candidate_mask,0)
        return self.from_vectors(cache[history_ids],history_ids.ne(0),cache[candidate_ids],candidate_mask,return_diagnostics)


def reciprocal_training_loss(output,click_labels,mode='rcrin_full',role_weight=.25,edge_weight=.20,pair_weight=.20):
    scores=output['scores'].float();mask=output['mask'];y=click_labels.float()
    # Masked labels are ignored even if caller fills padding with sentinel values.
    y=torch.where(mask,y,torch.zeros_like(y));count=mask.sum(1)
    active=count>0;denom=count.clamp_min(1)
    click=(F.binary_cross_entropy_with_logits(scores,y,reduction='none')*mask).sum(1)/denom
    pairmask=(mask&(y>0))[:,:,None]&(mask&(y<=0))[:,None,:]
    paircount=pairmask.sum((1,2)).clamp_min(1)
    pair=(F.softplus(-(scores[:,:,None]-scores[:,None,:]))*pairmask).sum((1,2))/paircount
    zero=click*0;role=zero;edge=zero
    if not mode.startswith('nrms_'):
        role=(F.binary_cross_entropy_with_logits(output['role_logits'].float(),y,reduction='none')*mask).sum(1)/denom
        edge=(F.softplus(-output['edge_logits'].float())*pairmask).sum((1,2))/paircount
    if mode=='nrms_bce':pair=zero
    per=click+role_weight*role+edge_weight*edge+pair_weight*pair
    total=(per*active).sum()/active.sum().clamp_min(1)
    return total,{'click':click.mean(),'role':role.mean(),'edge':edge.mean(),'pair':pair.mean()}
