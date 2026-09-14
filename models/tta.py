from collections import deque
import numpy as np
import torch
from torch import nn

class LatentAdapter(nn.Module):
    def __init__(self,dim=32,bottleneck=16):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,bottleneck),nn.GELU(),nn.Linear(bottleneck,dim));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,z): return z+self.net(z)

class ReliabilityGate:
    def __init__(self,tau,rho,kappa,window=5): self.tau,self.rho,self.kappa=tau,rho,kappa;self.history=deque(maxlen=window)
    def __call__(self,score,distance):
        self.history.append(float(score));stable=len(self.history)==self.history.maxlen and np.std(self.history)<=self.kappa
        return bool(score<=self.tau and distance<=self.rho and stable)

class ReliabilityConstrainedTTA:
    """Online adapter update performed only after the current point is scored."""
    def __init__(self,adapter,gate,reference,lr=1e-4):
        self.adapter,self.gate=adapter,gate
        self.reference=torch.as_tensor(reference,dtype=torch.float32)
        self.optimizer=torch.optim.Adam(adapter.parameters(),lr=lr)
        self.steps=0
    def process(self,z,pre_update_score):
        """Return pre-update adapted state and whether an update was accepted."""
        adapted=self.adapter(z)
        distance=torch.linalg.vector_norm(adapted.detach()-self.reference.to(z.device),dim=-1).mean()
        accepted=self.gate(float(pre_update_score),float(distance))
        if accepted:
            loss=((self.adapter(z)-z.detach())**2).mean()
            self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step()
            self.steps+=1
        return adapted.detach(),accepted
