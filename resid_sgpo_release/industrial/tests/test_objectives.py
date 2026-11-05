import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import objective

class Objectives(unittest.TestCase):
    def test_gspo_formula(self):
        lp=torch.tensor([[[-.3,-1.8],[-1.,-.2]]],requires_grad=True);old=lp.detach()-.02;adv=torch.tensor([[1.,-1.]])
        ratio=(lp-old).mean(-1).exp();expected=-torch.minimum(ratio*adv,ratio.clamp(.9,1.1)*adv).mean()
        self.assertTrue(torch.equal(expected,objective(lp,old,adv,'gspo',0,600)))
    def test_sgpo_routes_position_gradients(self):
        lp=torch.tensor([[[-.3,-1.8],[-1.,-.2]]],requires_grad=True);old=lp.detach();adv=torch.tensor([[1.,-1.]])
        objective(lp,old,adv,'sgpo',0,600).backward();self.assertTrue(torch.isfinite(lp.grad).all());self.assertFalse(torch.equal(lp.grad[:,:,0],lp.grad[:,:,1]))
    def test_zero_advantage(self):
        for method in ['grpo','gspo','sgpo']:
            lp=torch.zeros(2,8,6,requires_grad=True);loss=objective(lp,lp.detach(),torch.zeros(2,8),method,0,600);loss.backward();self.assertEqual(float(lp.grad.abs().sum()),0.)
if __name__=='__main__':unittest.main()
