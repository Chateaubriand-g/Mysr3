import lmdb
from io import BytesIO
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self,dataroot,datatype,lr,hr,split='train',datalen=-1,need_lr=False):
        self.dataroot = dataroot
        self.datatype = datatype
        self.lr = lr
        self.hr = hr
        self.split = split
        self.datalen = datalen
        self.need_lr = need_lr

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot,readonly=True,readahead=False,lock=False,meminit=False)
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get('length'.encode()))
            if self.datalen <= 0:
                self.datalen = self.dataset_len
            else:
                self.datalen = min(self.datalen,self.dataset_len)
        elif datatype == 'image':
            if self.need_lr:
                self.lr_path = [x for x in Path(dataroot)/'{}_lr'.format(lr) if x.suffix == '.png']
            self.hr_path = [x for x in Path(dataroot)/'{}_hr'.format(hr) if x.suffix == '.png']
            self.sr_path = [x for x in Path(dataroot)/'{}_{}_sr'.format(lr,hr) if x.suffix == '.png']
            self.dataset_len = len(self.hr_path)
            if self.datalen <= 0:
                self.datalen = self.dataset_len
            else:
                self.datalen = min(self.datalen,self.dataset_len)
        else:
            raise ValueError('datatype must be lmdb or image')
        
    def __getitem__(self,index):
        if self.datatype == 'lmdb':
            with self.env.begin(write = False) as txn:
                hr_img = txn.get('{}_hr_{}'.format(self.hr,str(index).zfill(5)).encode())
                sr_img = txn.get('{}_{}_sr_{}'.format(self.lr,self.hr,str(index).zfill(5)).encode())
                if self.need_lr:
                    lr_img = txn.get('{}_lr_{}'.format(self.lr,str(index).zfill(5)).encode())
                    lr_img = Image.open(BytesIO(lr_img)).convert('RGB')
                hr_img = Image.open(BytesIO(hr_img)).convert('RGB')
                sr_img = Image.open(BytesIO(sr_img)).convert('RGB')
        else:
            hr_img = Image.open(self.hr_path[index]).convert('RGB')
            sr_img = Image.open(self.sr_path[index]).convert('RGB')
            if self.need_lr:
                lr_img = Image.open(self.lr_path[index]).convert('RGB')
        if self.need_lr:
            return {'lr':lr_img,'hr':hr_img,'sr':sr_img,'index':index}
        else:
            return {'hr':hr_img,'sr':sr_img,'index':index}
        
    def __leng__(self):
        return self.datalen