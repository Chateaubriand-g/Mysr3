import torch
import math
from torch import nn
from config import DEVICE

class time_embedding(nn.Module):
    def __init__(self,time_dim):
        super().__init__()
        self.time_dim =time_dim

    def forward(self, t):
        half_time_dim = self.time_dim // 2
        emb = math.log(10000) / (half_time_dim - 1)
        emb = torch.exp(torch.arange(half_time_dim,device=DEVICE) * -emb)
        # t [batch_size,] emb [half_time_dim,]
        emb = t[:,None] * emb[None,:] # [batch_size, half_time_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # [batch_size, time_dim]
        assert emb.shape == (t.size(0), self.time_dim)
        return emb