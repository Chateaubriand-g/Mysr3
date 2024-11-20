import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='config/sr3_16_128.json',type=str,help='config file path')
    parser.add_argument('--pahse',type = str,defualt='train',help='train or test')
    parser.add_argument('--gpu_id',type = str,default = None,help='gpu id')
    parser.add_argument('--debug',action='store_true')
    parser.add_argument('--enable_wandb',action='store_true')
    parser.add_argument('-wandb_log_ckpt',action='store_true')
    parser.add_argument('--log_eval',action='store_true')

    args = parser.parse_args()

    