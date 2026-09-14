from torch import nn
from .state_space import NeuralStateSpace

class DCSS(nn.Module):
    def __init__(self,input_dim,config):
        super().__init__();m=config["model"];d=config["dependency"]
        self.backbone=NeuralStateSpace(input_dim,m["latent_dim"],m["hidden_dim"],m["history_dim"],m["dependency_context_dim"],d["max_lag"])
    def forward(self,x,dependency_messages): return self.backbone(x,dependency_messages)

