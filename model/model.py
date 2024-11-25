import os
from base_model import BaseModel
from torch.nn import Module
from torch.nn import functional as F

class DDPM(BaseModel):
    def __init__(self,opt):
        super().__init__(opt)
        self.net = self.device
