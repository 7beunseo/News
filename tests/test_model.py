import sys,unittest,inspect,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from model import RCRINRanker,ReciprocalCrossRoleInteraction,CrossRoleMessage,reciprocal_training_loss,MODES

torch.set_num_threads(2)

class RCRINTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(26111890)

    def sample(self):
        titles=torch.randint(1,60,(45,32));titles[0]=0
        h=torch.tensor([[1,2,0,0],[3,4,5,0],[0,0,0,0],[6,0,0,0]])
        ids=torch.tensor([[7,8,9,10],[11,12,13,0],[14,0,0,0],[0,0,0,0]])
        labels=torch.tensor([[1.,1.,0.,0.],[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,0.]])
        return titles,h,ids,ids.ne(0),labels

    def test_shapes_padding_singleton_empty_losses_and_no_label_input(self):
        titles,h,ids,mask,y=self.sample()
        for mode in MODES:
            net=RCRINRanker(60,mode=mode,relation_dim=16,heads=4,pair_chunk=2,query_chunk=2).eval()
            self.assertNotIn('labels',inspect.signature(net.forward).parameters)
            out=net(titles,h,ids,mask,True);self.assertEqual(out['scores'].shape,ids.shape)
            self.assertTrue(torch.isfinite(out['scores']).all());self.assertTrue((out['scores'][~mask]==0).all())
            for lab in (y,torch.zeros_like(y),mask.float()):
                loss,_=reciprocal_training_loss(out,lab,mode);self.assertTrue(torch.isfinite(loss))
            if net.relation is not None:
                self.assertTrue(torch.equal(out['scores'],out['base_logits'].masked_fill(~mask,0)))
                self.assertTrue(torch.equal(out['delta_plus'][2:],torch.zeros_like(out['delta_plus'][2:])))
                self.assertTrue(torch.equal(out['delta_minus'][2:],torch.zeros_like(out['delta_minus'][2:])))
                self.assertTrue(torch.equal(out['relation_delta'][2:],torch.zeros_like(out['relation_delta'][2:])))
            empty=net(titles,h,ids[:,:0],mask[:,:0]);loss,_=reciprocal_training_loss(empty,y[:,:0],mode)
            self.assertTrue(torch.isfinite(loss));self.assertEqual(float(loss),0.)

    def test_antisymmetry_permutation_chunking_and_padding(self):
        net=ReciprocalCrossRoleInteraction(12,16,4,2,pair_chunk=2,query_chunk=2).eval()
        torch.nn.init.normal_(net.score[-1].weight,std=.03)
        x=torch.randn(2,7,12);u=torch.randn(2,12);mask=torch.tensor([[1,1,1,1,0,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
        base=torch.randn(2,7);out=net(x,u,mask,base,True)
        self.assertTrue(torch.allclose(out['edges'],-out['edges'].transpose(1,2),atol=2e-6))
        self.assertTrue((out['edges'][0,4:]==0).all());self.assertTrue((out['edges'][0,:,4:]==0).all())
        self.assertTrue((out['edges'].diagonal(dim1=1,dim2=2)==0).all())
        perm=torch.tensor([5,1,6,2,4,0,3]);per=net(x[:,perm],u,mask[:,perm],base[:,perm],True)
        self.assertTrue(torch.allclose(per['scores'],out['scores'][:,perm],atol=3e-6,rtol=2e-5))
        net.pair_chunk=99;net.query_chunk=99;full=net(x,u,mask,base,True)
        self.assertTrue(torch.allclose(out['scores'],full['scores'],atol=3e-6,rtol=2e-5))
        self.assertTrue(torch.allclose(out['delta_plus'],full['delta_plus'],atol=3e-6,rtol=2e-5))
        padded=x.clone();padded[~mask]=1000
        changed=net(padded,u,mask,base,True)
        self.assertTrue(torch.allclose(changed['scores'],full['scores'],atol=3e-6))

    def test_null_role_response_and_parallel_updates(self):
        msg=CrossRoleMessage(16,4).eval();target=torch.randn(1,2,16);source=torch.randn(1,2,16)
        edges=torch.randn(1,2,2,16);mask=torch.ones(1,2,dtype=torch.bool)
        _,low=msg(target,source,edges,torch.full((1,2),.001),mask)
        _,high=msg(target,source,edges,torch.full((1,2),.999),mask)
        self.assertTrue((high>low).all())
        zero,mass=msg(target,source,edges,torch.ones(1,2)*.5,torch.tensor([[1,0]],dtype=torch.bool))
        self.assertTrue(torch.equal(zero[:,0],target[:,0]));self.assertEqual(float(mass.sum()),0.)
        net=ReciprocalCrossRoleInteraction(12,16,4,2).eval();records=[]
        def capture(name):
            def hook(module,args):records.append((name,args[0].detach().clone(),args[1].detach().clone()))
            return hook
        a=net.positive_reads_negative.register_forward_pre_hook(capture('p'))
        b=net.negative_reads_positive.register_forward_pre_hook(capture('n'))
        net(torch.randn(1,3,12),torch.randn(1,12),torch.ones(1,3,dtype=torch.bool),torch.zeros(1,3))
        a.remove();b.remove();self.assertEqual(len(records),4)
        for i in (0,2):
            self.assertTrue(torch.equal(records[i][1],records[i+1][2]))
            self.assertTrue(torch.equal(records[i][2],records[i+1][1]))

    def test_actual_gradients_and_label_isolation_cache(self):
        titles,h,ids,mask,y=self.sample();net=RCRINRanker(60,relation_dim=16,pair_chunk=2,query_chunk=2)
        opt=torch.optim.Adam(net.parameters(),lr=.001)
        for _ in range(2):
            opt.zero_grad();out=net(titles,h,ids,mask);loss,_=reciprocal_training_loss(out,y);loss.backward();opt.step()
        opt.zero_grad();out=net(titles,h,ids,mask);loss,_=reciprocal_training_loss(out,y);loss.backward()
        for name in ['words.weight','relation.role.0.weight','relation.pair.0.weight',
                     'relation.positive_reads_negative.q.weight','relation.negative_reads_positive.q.weight',
                     'relation.fusion_gate.0.weight','relation.interaction.weight','relation.score.0.weight']:
            p=dict(net.named_parameters())[name];self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name);self.assertGreater(float(p.grad.norm()),0.,name)
        net.eval()
        with torch.no_grad():
            before=net(titles,h,ids,mask)['scores'];other=net(titles,h,ids,mask)
            reciprocal_training_loss(other,1-y)
            after=net(titles,h,ids,mask)['scores'];self.assertTrue(torch.equal(before,after))
            cache=net.encode_news(titles);cached=net.cached(cache,h,ids,mask)['scores']
            self.assertTrue(torch.allclose(before,cached,atol=3e-6,rtol=2e-5))

    def test_anchor_and_auxiliary_only_ablation(self):
        net=ReciprocalCrossRoleInteraction(12,16,4,2,variant='rcrin_no_exchange').eval()
        torch.nn.init.normal_(net.score[-1].weight,std=.2);torch.nn.init.constant_(net.score[-1].bias,1.)
        out=net(torch.randn(2,3,12),torch.randn(2,12),torch.ones(2,3,dtype=torch.bool),torch.randn(2,3))
        self.assertTrue(torch.equal(out['relation_delta'],torch.zeros_like(out['relation_delta'])))

    def test_chunk_backward_parity_and_shared_initial_backbone(self):
        a=ReciprocalCrossRoleInteraction(12,16,4,2,pair_chunk=2,query_chunk=2).train()
        torch.nn.init.normal_(a.score[-1].weight,std=.03);b=copy.deepcopy(a);b.pair_chunk=99;b.query_chunk=99;b.pair_checkpoint=False
        x=torch.randn(1,5,12);u=torch.randn(1,12);mask=torch.ones(1,5,dtype=torch.bool);base=torch.zeros(1,5);y=torch.tensor([[1.,0.,1.,0.,0.]])
        for net in (a,b):
            out=net(x,u,mask,base);loss,_=reciprocal_training_loss(out,y);loss.backward()
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.grad is not None:self.assertTrue(torch.allclose(p.grad,q.grad,atol=3e-6,rtol=2e-4),name)
        torch.manual_seed(42);first=RCRINRanker(60,'nrms_bce')
        torch.manual_seed(42);second=RCRINRanker(60,'rcrin_full')
        self.assertTrue(all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items()))

if __name__=='__main__':unittest.main(verbosity=2)
