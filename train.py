import argparse
import myutils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='config/sr3_16_128.json',type=str,help='config file path')
    parser.add_argument('--phase',type = str,default='train',help='train or test')
    parser.add_argument('--gpu_id',type = str,default = None,help='gpu id')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--enable_wandb',action='store_true')
    parser.add_argument('-wandb_log_ckpt',action='store_true')
    parser.add_argument('--log_eval',action='store_true')

    #设置config
    args = parser.parse_args()
    opt = myutils.parse(args)
    opt = myutils.dict_to_nonedict(opt)

    #dataloader
    for pahse,dataset_opt in opt['datasets'].item():
        if pahse == 'train' and args.phase == 'train':
            dataset = myutils.create_dataset(dataset_opt,pahse)
            dataloader = myutils.create_dataloader(dataset,dataset_opt,pahse)
        elif pahse == 'test' and args.phase == 'test':
            dataset = myutils.create_dataset(dataset_opt,pahse)
            dataloader = myutils.create_dataloader(dataset,dataset_opt,pahse)
        else:
            raise NotImplementedError('phase [%s] is not implemented'.format(pahse))
    