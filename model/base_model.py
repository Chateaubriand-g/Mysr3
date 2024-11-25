import torch


class BaseModel():
    def __init__(self,opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if isinstance(x,dict):
            for key,value in x.items():
                x[key] = value.to(self.device)
        elif isinstance(x,list):
            [y.to(self.device) for y in x if y is not None]
        else:
            x.to(self.device)
        return x
    
    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network,torch.nn.DataParallel):
            network = network.module
        s = str(network)
        # n = sum(map(lambda x: x.numel(), network.parameters()))
        n = sum(parameters.numel() for parameters in network.parameters())
        return s, n
        