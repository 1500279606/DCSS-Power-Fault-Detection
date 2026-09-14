import torch
from torch import nn
import torch.nn.functional as F

class NeuralStateSpace(nn.Module):
    def __init__(self,input_dim,latent=32,hidden=64,history=128,dependency_dim=64,lag=2):
        super().__init__();self.lag=lag
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Linear(hidden,latent))
        self.history=nn.GRU(latent,history,batch_first=True)
        self.dep=nn.Sequential(nn.LazyLinear(hidden),nn.GELU(),nn.Linear(hidden,dependency_dim))
        self.base=nn.Sequential(nn.Linear(latent+history,hidden),nn.GELU(),nn.Linear(hidden,latent))
        self.dep_proj=nn.Linear(dependency_dim,latent);self.gate=nn.Linear(latent+dependency_dim,latent)
        self.decoder=nn.Sequential(nn.Linear(latent,hidden),nn.GELU(),nn.Linear(hidden,input_dim))
        self.log_q=nn.Parameter(torch.full((latent,),-2.25));self.log_r=nn.Parameter(torch.full((input_dim,),-2.25))
        self.update=nn.Sequential(nn.Linear(latent*2+input_dim,hidden),nn.GELU(),nn.Linear(hidden,latent))
    def forward(self,x,dependency_messages):
        z=self.encoder(x);h,_=self.history(z[:,:-1]);prev=z[:,self.lag-1:-1];h=h[:,self.lag-1:]
        c=self.dep(dependency_messages);base=self.base(torch.cat([prev,h],-1));causal=self.dep_proj(c)
        gamma=torch.sigmoid(self.gate(torch.cat([prev,c],-1)));prior=prev+base+gamma*causal
        pred=self.decoder(prior);obs=x[:,self.lag:];var=F.softplus(self.log_r)+1e-5
        normalized=(obs-pred)/torch.sqrt(var);posterior=prior+self.update(torch.cat([prior,torch.log(F.softplus(self.log_q)+1e-5).expand_as(prior),normalized],-1))
        nll=.5*(torch.log(var)+(obs-pred).square()/var+torch.log(torch.tensor(2*torch.pi,device=x.device))).mean()
        return {"encoded":z,"prior":prior,"posterior":posterior,"prediction":pred,"reconstruction":self.decoder(z),"nll":nll,"gate":gamma,"base_delta":base,"dependency_delta":causal}

