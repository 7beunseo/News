import copy,inspect,unittest
import numpy as np
import torch
from torch.nn import functional as F
from role_free.model import AntisymmetricRelation,RoleFreeRanker,training_loss,potential_residual_ratio
from role_free.experiment import sample_columns,new_model,seed_all,state_hash


class RelationTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(71);torch.set_num_threads(2)

    def test_factorization_values_and_gradients(self):
        m=AntisymmetricRelation(8).double();ref=copy.deepcopy(m)
        args=[torch.randn(*shape,dtype=torch.double,requires_grad=True) for shape in ((2,5,8),(2,8),(2,8))]
        args2=[x.detach().clone().requires_grad_(True) for x in args]
        y=torch.cat([m.block(*args,i,min(i+2,5)) for i in range(0,5,2)],1);z=ref.dense_reference(*args2)
        torch.testing.assert_close(y,z,atol=1e-12,rtol=1e-10)
        weight=torch.randn_like(y);(y*weight).sum().backward();(z*weight).sum().backward()
        for a,b in zip([*args,*m.parameters()],[*args2,*ref.parameters()]):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-11,rtol=1e-9)
        torch.testing.assert_close(y,-y.transpose(1,2),atol=1e-12,rtol=1e-10)

    def example(self,mode='rf_full'):
        m=RoleFreeRanker(100,mode,relation_dim=16,heads=4,rounds=2,pair_chunk=2,news_chunk=24).eval()
        with torch.no_grad():m.head[-1].weight.normal_(std=.03)
        h=torch.randn(2,3,256);hm=torch.ones(2,3,dtype=torch.bool);c=torch.randn(2,5,256)
        mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
        return m,h,hm,c,mask

    def test_equivariance_padding_and_antisymmetry(self):
        m,h,hm,c,mask=self.example();a=m.from_vectors(h,hm,c,mask)
        perm=torch.tensor([3,0,4,1,2]);b=m.from_vectors(h,hm,c[:,perm],mask[:,perm])
        torch.testing.assert_close(a['scores'][:,perm],b['scores'],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(a['edge_logits'][:,perm][:,:,perm],b['edge_logits'],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(a['edge_logits'],-a['edge_logits'].transpose(1,2),atol=1e-6,rtol=1e-5)
        cp=c.clone();cp[~mask]=1000.;z=m.from_vectors(h,hm,cp,mask)
        torch.testing.assert_close(a['scores'],z['scores'])
        more=torch.cat([c,torch.randn(2,3,256)*100],1);mm=torch.cat([mask,torch.zeros(2,3,dtype=torch.bool)],1)
        z=m.from_vectors(h,hm,more,mm)
        torch.testing.assert_close(a['scores'],z['scores'][:,:5],atol=1e-5,rtol=1e-5)

    def test_singleton_empty_and_aux_identity(self):
        for mode in ('rf_full','rf_no_relation_loss','plain_attention_pair','rf_aux_only','nrms_ce_pair'):
            m,h,hm,c,mask=self.example(mode)
            for n in (0,1):
                out=m.from_vectors(h,hm,c[:,:n],mask[:,:n])
                self.assertTrue(torch.isfinite(out['scores']).all())
                torch.testing.assert_close(out['relation_delta'],torch.zeros_like(out['relation_delta']))
            out=m.from_vectors(h,hm,c,torch.zeros_like(mask));self.assertEqual(float(out['scores'].abs().sum()),0.)
            if mode=='rf_aux_only':
                out=m.from_vectors(h,hm,c,mask);self.assertEqual(float(out['relation_delta'].abs().sum()),0.)

    def test_loss_only_cross_label_pairs(self):
        score=torch.tensor([[.2,.1,-.5,9.]],requires_grad=True)
        rho=torch.randn(1,4,4,requires_grad=True);mask=torch.tensor([[1,1,1,0]],dtype=torch.bool)
        y=torch.tensor([[1.,1.,0.,1.]])
        owners=torch.tensor([0,0]);columns=torch.tensor([[0,2,2,2,2],[1,2,2,2,2]])
        out=dict(scores=score,mask=mask,edge_logits=rho)
        loss,parts=training_loss(out,y,owners,columns,'rf_full')
        expected=F.cross_entropy(score[owners[:,None],columns],torch.zeros(2,dtype=torch.long))+.2*F.softplus(-rho[0,:2,2]).mean()
        torch.testing.assert_close(loss,expected);loss.backward()
        allowed=torch.zeros_like(rho,dtype=torch.bool);allowed[0,:2,2]=True
        self.assertEqual(float(rho.grad[~allowed].abs().sum()),0.);self.assertTrue((rho.grad[allowed]<0).all())

    def test_forward_no_labels_and_cache_parity(self):
        self.assertNotIn('labels',inspect.signature(RoleFreeRanker.forward).parameters)
        m,_,_,_,_=self.example();titles=torch.randint(1,100,(15,8));titles[0]=0
        h=torch.tensor([[1,2,0],[3,0,0]]);c=torch.tensor([[4,5,6],[7,8,0]]);mask=c>0
        out=m(titles,h,c,mask)
        with torch.no_grad():
            cache=m.encode_news(titles);cache[0]=0
            cached=m.cached(cache,h,c,mask)
        torch.testing.assert_close(out['scores'],cached['scores'],atol=2e-5,rtol=1e-5)

    def test_gradient_paths(self):
        m=RoleFreeRanker(100,'rf_full',16,4,2,2,24).train();opt=torch.optim.Adam(m.parameters(),lr=.01)
        titles=torch.randint(1,100,(15,8));titles[0]=0;h=torch.tensor([[1,2,0]]);c=torch.tensor([[3,4,5]]);mask=c>0
        y=torch.tensor([[1.,0.,0.]]);owner=torch.tensor([0]);cols=torch.tensor([[0,1,2,1,2]])
        for _ in range(3):
            opt.zero_grad();out=m(titles,h,c,mask);loss,_=training_loss(out,y,owner,cols,'rf_full');loss.backward();opt.step()
        for module in (m.words,m.relation,m.edge_score,m.attention,m.head):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0.)

    def test_potential_projection_diagnostic(self):
        v=torch.tensor([[1.,3.,-2.]]);r=v[:,:,None]-v[:,None,:];mask=torch.ones(1,3,dtype=torch.bool)
        ratio,_=potential_residual_ratio(r,mask);self.assertLess(float(ratio),1e-10)
        cycle=torch.tensor([[[0.,1.,-1.],[-1.,0.,1.],[1.,-1.,0.]]]);ratio,_=potential_residual_ratio(cycle,mask)
        self.assertAlmostEqual(float(ratio),1.,places=6)

    def test_matched_initialization_and_sampler(self):
        hashes=[]
        for mode in ('rf_full','rf_no_relation_loss','plain_attention_pair','nrms_ce_pair','rf_aux_only'):
            seed_all();hashes.append(state_hash(new_model(100,mode)))
        self.assertEqual(len(set(hashes)),1)
        class Fake:
            a=dict(offsets=np.array([0,4,7]),train_labels=np.array([1,0,1,0,0,1,0]))
        owners,cols=sample_columns(Fake(),np.array([0,1]),np.random.default_rng(2))
        self.assertEqual(owners.tolist(),[0,0,1]);self.assertEqual(cols[:,0].tolist(),[0,2,1])
        self.assertTrue(np.isin(cols[:2,1:],[1,3]).all());self.assertTrue(np.isin(cols[2,1:],[0,2]).all())


if __name__=='__main__':unittest.main(verbosity=2)
