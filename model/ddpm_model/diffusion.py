import math
import torch
from inspect import isfunction


def warmup_beta(beta_start,beta_end,steps,warmup_ration):
    beta = beta_end*torch.ones(steps,dtype=torch.float64)
    assert steps*warmup_ration<=steps,"warmup steps should be less than total steps"
    warmup_steps = int(steps*warmup_ration)
    beta[:warmup_steps] = torch.linspace(beta_start,beta_end,warmup_steps,dtype=torch.float64)
    return beta


def get_beta_schedule(schedule,steps,beta_start=1e-4,beta_end=2e-2,cosine=8e-3):
    """
    schedule: 'quad' beta_start**0.5,beta_end**0.5,beta**2
              'linear' beta_start,beta_end,beta
              'const' (beta_end,beta_end,...,beta_end)
              'jsd' (1/T,1/T-1,1/T-2,...,1)
              'warmup_10', 'warmup_50', 'cosine'
    """
    if schedule == 'quad':
        beta = torch.linspace(beta_start**0.5,beta_end**0.5,steps, dtype=torch.float64)**2
    elif schedule == 'linear':
        beta = torch.linspace(beta_start,beta_end,steps, dtype=torch.float64)
    elif schedule == 'const':
        beta = beta_end*torch.ones(steps,dtype=torch.float64)
    elif schedule == 'jsd':
        beta = 1.0/torch.linspace(steps,1,steps,dtype=torch.float64)
    elif schedule == 'warmup_10':
        beta = warmup_beta(beta_start,beta_end,steps,0.1)
    elif schedule == 'warmup_50':
        beta = warmup_beta(beta_start,beta_end,steps,0.5)
    elif schedule == 'cosine':
        temp = torch.arange(steps,dtype=torch.float64)/steps+cosine
        temp = (temp/(1+cosine)*math.pi/2)
        temp = temp.cos().pow(2)
        alpha_comprod = temp/temp[0]
        alpha_comprod_pred = torch.cat(torch.ones(1,dtype=torch.float64),alpha_comprod[:-1])
        beta = 1-alpha_comprod/alpha_comprod_pred
    else:
        raise NotImplementedError(f'unknown beta schedule: {schedule}')
    return beta


def exists(x):
    return x is not None


def default(val,d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(alpha_cumprod,t,x_shape):
    batch_size,_ = t.shape
    output = alpha_cumprod.gather(-1,t)
    return output.reshape(batch_size,*([1]*len(x_shape[:-1]))) # [batch_size,1,1,1,...]


class Diffusion(torch.nn.Module):
    def __init__(self,denoise_fn,image_size,channels=3,loss_type='l1',conditional=True,schedule_opt=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.image_size = image_size
        self.channels = channels
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is None:
            pass

    def set_loss(self,device):
        if self.loss_type == 'l1':
            self.loss_fn = torch.nn.L1Loss(reduciton='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_fn = torch.nn.MSELoss(reduciton='sum').to(device)

    def set_noise_schedule(self,schedule_opt,device):
        beta = get_beta_schedule(schedule=schedule_opt['schedule'],
                                 steps=schedule_opt['num_steps'],
                                 beta_start=schedule_opt['start'],
                                 beta_end=schedule_opt['end'])
        alpha = 1.0-beta
        alpha_cumprod = torch.cumprod(alpha,dim=0)
        alpha_cumprod_prev = torch.cat(torch.ones(1,dtype=torch.float64),alpha_cumprod[:-1])
        steps = schedule_opt['num_steps']
        
        #diffusion
        self.register_buffer('sqrt_alphas_cumprod',torch.sqrt(alpha_cumprod).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',torch.sqrt(1.0-alpha_cumprod).to(device))
        #denoise
        self.register_buffer('variance',torch.sqrt(beta(1-alpha_cumprod_prev)/(1-alpha_cumprod)).to(device))
        # self.register_buffer('mean_coef1',1.0/torch.sqrt(alpha).to(device))
        # self.register_buffer('mean_coef2',(1-alpha).to(device)/torch.sqrt(1.0-alpha_cumprod).to(device))
        self.register_buffer('mean_coef1',torch.sqrt(alpha).to(device)*(1-alpha_cumprod_prev).to(device)/(1-alpha_cumprod).to(device))
        self.register_buffer('mean_coef2',torch.sqrt(alpha_cumprod_prev).to(device)*(1-alpha).to(device)/(1-alpha_cumprod).to(device))
        

     #计算前向过程的均值和方差
    def diffusion_mean_variance(self,x_0,t):
        mean = extract(self.sqrt_alphas_cumprod,t,x_0.shape)
        variance = extract(self.sqrt_one_minus_alpha_cumprod,t,x_0.shape)
        return mean,variance
    
    def predict_x0_from_noise(self,x_t,t,noise):
        x_0 = (x_t-extract(self.sqrt_one_minus_alpha_cumprod,t,x_t.shape)*noise)/extract(self.sqrt_alphas_cumprod,t,x_t.shape)
        return x_0
    
    def posterior_mean_variance(self,x_0,x_t,t):
        mean = extract(self.mean_coef1,t,x_t.shape)*x_t+extract(self.mean_coef2,t,x_0.shape)*x_0
        variance = extract(self.variance,t,x_t.shape)
        return mean,variance
    
    def predict_mean_variance(self,x,t,clip=False,condition=None):
        if condition:
            x_0 = self.predict_x0_from_noise(x,t,noise=self.denoise_fn(torch.cat([x,condition],dim=1),t))
        else:
            x_0 = self.predict_x0_from_noise(x,t,noise=self.denoise_fn(x,t))
        
        if clip:
            x_0 = torch.clamp(x_0-1,1)

        mean,variance = self.posterior_mean_variance(x_0,x,t)
        return mean,variance
    
    
        
