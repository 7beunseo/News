"""Paper-based NRMS baseline, trained end-to-end from random initialization.

Architecture reference: Wu et al., EMNLP 2019, DOI 10.18653/v1/D19-1671.
Cross-checked with Recommenders nrms.py / SelfAttention / AttLayer2 (MIT).
This is a PyTorch reimplementation, not execution of the original TF repository.
Explicit differences: random word initialization, padding masked in both pools,
and no GloVe. No learned user/item ID table, old backbone, or score input.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MultiHead(nn.Module):
    def __init__(self, input_dim, heads=16, head_dim=16):
        super().__init__(); self.heads=heads; self.head_dim=head_dim
        self.q=nn.Linear(input_dim,heads*head_dim,bias=False)
        self.k=nn.Linear(input_dim,heads*head_dim,bias=False)
        self.v=nn.Linear(input_dim,heads*head_dim,bias=False)
        for m in (self.q,self.k,self.v): nn.init.xavier_uniform_(m.weight)
    def forward(self,x,mask):
        b,n,_=x.shape
        q,k,v=[m(x).view(b,n,self.heads,self.head_dim).transpose(1,2) for m in (self.q,self.k,self.v)]
        safe=mask.clone(); safe[:,0] |= ~safe.any(1)
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=safe[:,None,None,:],dropout_p=0)
        return y.transpose(1,2).reshape(b,n,-1)*mask[:,:,None]

class AdditivePool(nn.Module):
    def __init__(self,dim=256,hidden=200):
        super().__init__(); self.affine=nn.Linear(dim,hidden);self.query=nn.Linear(hidden,1,bias=False)
        nn.init.xavier_uniform_(self.affine.weight);nn.init.zeros_(self.affine.bias)
        nn.init.xavier_uniform_(self.query.weight)
    def forward(self,x,mask):
        logits=self.query(torch.tanh(self.affine(x))).squeeze(-1).float()
        safe=mask.clone();safe[:,0] |= ~safe.any(1)
        w=logits.masked_fill(~safe,-1e9).softmax(-1)*mask
        return (x*w.to(x.dtype)[:,:,None]).sum(1)

class NRMS(nn.Module):
    def __init__(self,vocab_size,dropout=.2):
        super().__init__();self.dim=256
        self.words=nn.Embedding(vocab_size,300,padding_idx=0)
        nn.init.normal_(self.words.weight,mean=0,std=.05)
        with torch.no_grad(): self.words.weight[0].zero_()
        self.word_attention=MultiHead(300);self.word_pool=AdditivePool()
        self.user_attention=MultiHead(256);self.user_pool=AdditivePool()
        self.dropout=nn.Dropout(dropout)
    def encode_news(self,titles):
        mask=titles.ne(0);x=self.dropout(self.words(titles))
        x=self.dropout(self.word_attention(x,mask))
        return self.word_pool(x,mask)
    def encode_user(self,news,mask):
        return self.user_pool(self.user_attention(news,mask),mask)
    def forward(self,titles,history_ids,candidate_ids):
        # Encode only nonpadding occurrences. No cross-step detached train cache,
        # no unique-item dedup: duplicate occurrences retain separate dropout draws.
        b,h=history_ids.shape; k=candidate_ids.shape[1]
        ids=torch.cat([history_ids,candidate_ids],1); flat=ids.flatten(); present=flat.ne(0)
        enc=self.encode_news(titles[flat[present]])
        output=torch.zeros((len(flat),self.dim),device=enc.device,dtype=enc.dtype)
        output[present]=enc; output=output.view(b,h+k,self.dim)
        user=self.encode_user(output[:,:h],history_ids.ne(0))
        return (output[:,h:]*user[:,None,:]).sum(-1).float()
