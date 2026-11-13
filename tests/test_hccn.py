import copy,inspect,unittest
import numpy as np
import torch
from torch.nn import functional as F
from hccn.model import HCCN,PairGraph,choice_readout,masked_softmax
from hccn.experiment import sampled_ce,candidates_for,loss_columns


class HCCNTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(83);torch.set_num_threads(2)
    def model(self,mode='hccn_full'):
        return HCCN(100,mode,word_dim=24,dim=32,news_layers=2,heads=4,feedforward=64,
            slots=4,relation_dim=8,relation_hidden=16,dropout=0,pair_chunk=2,news_chunk=24,title_tokens=8)
    def example(self):
        return torch.randn(2,4,32),torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool),torch.randn(2,5,32),torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
    def activate(self,m):
        with torch.no_grad():m.graph.rho.weight.normal_(std=.1);m.graph.competition[-1].weight.normal_(std=.1)

    def test_neutral_initialization_equals_centered_evidence(self):
        m=self.model().eval();h,hm,z,mask=self.example();o=m.from_vectors(h,hm,z,mask,True)
        centered=(o['evidence']-o['evidence'].sum(1,keepdim=True)/mask.sum(1,keepdim=True))*mask
        torch.testing.assert_close(o['scores'],centered,atol=2e-7,rtol=2e-6)
        cols=torch.tensor([[0,1,2],[2,0,1]])
        torch.testing.assert_close(F.cross_entropy(o['scores'].gather(1,cols),torch.zeros(2,dtype=torch.long)),
            F.cross_entropy(o['evidence'].gather(1,cols),torch.zeros(2,dtype=torch.long)))
        self.assertEqual(float(o['rho'].abs().sum()),0.)

    def test_readout_gradient_and_padding_count(self):
        a=torch.randn(2,4,dtype=torch.double,requires_grad=True);mask=torch.tensor([[1,1,1,1],[1,1,0,0]],dtype=torch.bool)
        score,_,_=choice_readout(a,torch.zeros(2,4,4,dtype=torch.double),torch.zeros(2,4,4),mask,0,4)
        reference=(a-(a*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True))*mask
        torch.testing.assert_close(score,reference,atol=1e-7,rtol=1e-7)
        w=torch.randn_like(a)
        torch.testing.assert_close(torch.autograd.grad((score*w).sum(),a,retain_graph=True)[0],
            torch.autograd.grad((reference*w).sum(),a)[0],atol=1e-7,rtol=1e-7)

    def test_pair_symmetry_antisymmetry_and_exact_factorization(self):
        g=PairGraph(16,8,16).double();z=torch.randn(1,5,16,dtype=torch.double,requires_grad=True)
        memory=torch.randn(1,4,16,dtype=torch.double,requires_grad=True);mask=torch.ones(1,5,dtype=torch.bool)
        p=g.prepare(z,memory,mask);_,_,alpha,interest,r=g.block(p,0,5)
        torch.testing.assert_close(alpha,alpha.transpose(1,2));torch.testing.assert_close(interest,interest.transpose(1,2))
        torch.testing.assert_close(r,-r.transpose(1,2),atol=1e-12,rtol=1e-10)
        ref=g.dense_relation_reference(p,interest);torch.testing.assert_close(r,ref,atol=1e-12,rtol=1e-9)
        w=torch.randn_like(r);parameters=[z,memory,*g.first.parameters(),*g.last.parameters()]
        grad1=torch.autograd.grad((r*w).sum(),parameters,retain_graph=True)
        grad2=torch.autograd.grad((ref*w).sum(),parameters)
        for a,b in zip(grad1,grad2):torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-8)

    def test_permutation_padding_chunk_parity(self):
        m=self.model().eval();self.activate(m);h,hm,z,mask=self.example();a=m.from_vectors(h,hm,z,mask,True)
        perm=torch.tensor([3,0,4,1,2]);b=m.from_vectors(h,hm,z[:,perm],mask[:,perm],True)
        torch.testing.assert_close(a['scores'][:,perm],b['scores'],atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(a['rho'][:,perm][:,:,perm],b['rho'],atol=2e-6,rtol=2e-5)
        zp=z.clone();zp[~mask]=1e4
        torch.testing.assert_close(a['scores'],m.from_vectors(h,hm,zp,mask)['scores'])
        extra=torch.cat([z,torch.randn(2,2,32)*100],1);mm=torch.cat([mask,torch.zeros(2,2,dtype=torch.bool)],1)
        torch.testing.assert_close(a['scores'],m.from_vectors(h,hm,extra,mm)['scores'][:,:5],atol=2e-6,rtol=2e-5)
        m.pair_chunk=5;torch.testing.assert_close(a['scores'],m.from_vectors(h,hm,z,mask)['scores'],atol=2e-6,rtol=2e-5)

    def test_singleton_empty_history_titles_and_candidates(self):
        for mode in ('hccn_full','common_interest','individual','self_attention','no_rho'):
            m=self.model(mode).eval();h,hm,z,mask=self.example()
            for n in (0,1,5):
                out=m.from_vectors(h,torch.zeros_like(hm),z[:,:n],mask[:,:n],True)
                self.assertTrue(torch.isfinite(out['scores']).all())
                if n==1:torch.testing.assert_close(out['scores'],out['evidence'])
                self.assertEqual(float(out['memory'].abs().sum()),0.)
            enc=m.encode_news(torch.zeros(2,8,dtype=torch.long));self.assertEqual(float(enc.abs().sum()),0.)

    def test_ce_gradients_reach_all_core_modules(self):
        m=self.model().train();opt=torch.optim.Adam(m.parameters(),lr=.005)
        titles=torch.randint(1,100,(12,8));titles[0]=0;h=torch.tensor([[1,2,3,4],[4,3,2,1]]);c=torch.tensor([[5,6,7],[7,8,9]]);mask=c>0
        cols=torch.tensor([[0,1,2,1,2],[1,0,2,0,2]])
        for _ in range(4):
            opt.zero_grad();o=m(titles,h,c,mask);loss=sampled_ce(o['scores'],cols,2);loss.backward();opt.step()
        for name,module in [('news',m.news),('memory',m.memory),('evidence',m.evidence),('pair_query',m.graph.pair_query),
            ('relation',m.graph.first),('rho',m.graph.rho),('competition',m.graph.competition)]:
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0.,name)

    def test_no_outcome_forward_and_cached_parity(self):
        self.assertNotIn('labels',inspect.signature(HCCN.forward).parameters)
        m=self.model().eval();self.activate(m);titles=torch.randint(1,100,(12,8));titles[0]=0
        h=torch.tensor([[1,2,0],[3,4,0]]);c=torch.tensor([[5,6,7],[7,8,0]]);mask=c>0
        with torch.no_grad():
            out=m(titles,h,c,mask);cache=m.encode_news(titles);cached=m.cached(cache,h,c,mask)
        torch.testing.assert_close(out['scores'],cached['scores'],atol=2e-6,rtol=2e-5)

    def test_common_interest_no_rho_and_interventions(self):
        m=self.model().eval();self.activate(m);h,hm,z,mask=self.example();p=m.graph.prepare(z,m.memory(h,hm),mask)
        _,_,_,common,_=m.graph.block(p,0,5,True)
        torch.testing.assert_close(common,common[:,:1,:1].expand_as(common))
        _,_,_,pair,_=m.graph.block(p,0,5,False);self.assertGreater(float((pair-pair[:,:1,:1]).abs().sum()),1e-8)
        out=m.from_vectors(h,hm,z,mask,True,intervention='permute_pair_context')
        self.assertTrue(torch.isfinite(out['scores']).all())
        m.mode='no_rho';out=m.from_vectors(h,hm,z,mask,True);self.assertEqual(float(out['rho'].abs().sum()),0.)
        m.train()
        with self.assertRaises(AssertionError):m.from_vectors(h,hm,z,mask,intervention='common')

    def test_positive_example_mean_and_sampling(self):
        class Fake:
            a=dict(positive_origins=np.array([0,0,1]),positive_items=np.array([10,12,15]),fit_mask=np.array([1,1],bool),
                offsets=np.array([0,4,7]),items=np.arange(10,17),train_labels=np.array([1,0,1,0,0,1,0]))
        qq,cc=candidates_for(Fake(),np.array([0,1,2]),np.random.default_rng(7));cols=loss_columns(Fake(),qq,cc)
        self.assertEqual(qq.tolist(),[0,0,1]);self.assertEqual(cols[:,0].tolist(),[0,2,1])
        scores=torch.randn(3,4,requires_grad=True);cols=torch.tensor(cols)
        loss=sampled_ce(scores,cols,3)
        manual=sum(F.cross_entropy(scores[i:i+1].gather(1,cols[i:i+1]),torch.zeros(1,dtype=torch.long)) for i in range(3))/3
        torch.testing.assert_close(loss,manual)


if __name__=='__main__':unittest.main(verbosity=2)
