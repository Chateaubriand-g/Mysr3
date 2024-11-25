import os
import json
from collections import OrderedDict
from datetime import datetime


def mkdirs(paths):
    if isinstance(paths,str):
        os.makedirs(paths,exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path,exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def parse(args):
    phase =args.phase
    config_path = args.config
    gpu_ids = args.gpu_ids
    enable_wandb = args.enable_wandb

    #删除注释并解析json文件
    json_str = ''
    with open(config_path,'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict) # OrderedDict保证json文件中键的顺序不变

    #设置log文件路径
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    experiments_root = os.path.join('experiments','{}_{}'.format(opt['name'],get_timestamp()))
    opt['path']['experiment_root'] = experiments_root
    for key,path in opt['path'].items():
        if 'resume' not in key or 'expreiments' not in key:
            opt['path'][key] = os.path.join(experiments_root,path)
            mkdirs(opt['path'][key])

    opt['phase'] = phase

    #设置分布式gpu
    if gpu_ids is not None:
        opt['gpu_ids'] = [int(x) for x in gpu_ids.split(',')]
        gpu_list = gpu_ids
    else:
        gpu_list = ','.join([str(x) for x in opt['gpu_ids']])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    #设置wandb
    try:
        wandb_log_checkpoint = args.wandb_log_checkpoint
        opt['wandb']['wandb_log_checkpoint'] = wandb_log_checkpoint
    except:
        pass
    try:
        wandb_log_eval = args.wandb_log_eval
        opt['wandb']['wandb_log_eval'] = wandb_log_eval
    except:
        pass
    try:
        wandb_log_info = args.wandb_log_info
        opt['wandb']['wandb_log_info'] = wandb_log_info
    except:
        pass
    opt['wandb']['enable_wandb'] = enable_wandb

    #设置nonedict类，当未匹配到键对时返回None而不是抛出异常
    
class NoneDict(dict):
    def __missing__(self, key):
        return None
    
#将opt字典中的所有dict转换为Nonedict类
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key,val in opt.items():
            new_opt[key] = dict_to_nonedict(val)
        return NoneDict(**new_opt)
    elif isinstance(opt,list):
        return [dict_to_nonedict(x) for x in opt]
    else:
        return opt
