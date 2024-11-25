#对图像文件进行预处理
import lmdb
import multiprocessing
import numpy as np
from argparse import ArgumentParser
from multiprocessing import sharedctypes
from functools import partial
from pathlib import Path
from io import BytesIO
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from torchvision.transforms import functional as TF


def resize_and_convert(img,size,resample):
    if img.size[0] != size:
        img = TF.resize(img,size,resample)
        img = TF.center_crop(img,size)
    return img

def image_convert_bytes(img):
    b = BytesIO()
    img.save(b,format='PNG')
    return b.getvalue()

def resize_image(img,size,resample,lmdb_save):
    lr_img = resize_and_convert(img,size[0],resample)
    hr_img = resize_and_convert(img,size[1],resample)
    sr_img = resize_and_convert(lr_img,size[1],resample)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    return [lr_img,hr_img,sr_img]

def resize_worker(img_file,size,resample,lmdb_save):
    img = Image.open(img_file)
    img = img.convert('RGB')
    img = resize_image(img,size,resample,lmdb_save)
    return img_file.name.split('.')[0],img

class WorkingContext():
    def __init__(self,resize_fn,lmdb_save,out_path,env,size):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.size = size

        self.counter = sharedctypes.RawValue('i',0)
        self.lock = multiprocessing.Lock()

    def increment_counter(self):
        with self.lock:
            self.counter.value += 1
            return self.counter.value
        
    def value(self):
        with self.lock:
            return self.counter.value
        
def prepare_process(img_subset,wctx:WorkingContext):
    for image in img_subset:
        i,img = wctx.resize_fn(image)
        lr_img,hr_img,sr_img = img

        if not wctx.lmdb_save:
            lr_img.save('{}/{}_lr/{}.png'.format(wctx.out_path,wctx.size[0],str(i).zfill(5))) #zfill函数将字符串补齐到指定长度，右对齐,i应为字符串，所以加上str()增加鲁棒性
            hr_img.save('{}/{}_hr/{}.png'.format(wctx.out_path,wctx.size[1],str(i).zfill(5)))
            sr_img.save('{}/{}_{}_sr/{}.png'.format(wctx.out_path,wctx.size[0],wctx.size[1],str(i).zfill(5)))
        else:
            with wctx.env.begin(write = True) as txn:  #lmdb事务的开启，在其他函数已经实现了lmdb库的创建,在lmdb中，同时只能有一个写事务
                txn.put('{}_lr_{}'.format(wctx.size[0],str(i).zfill(5)).encode(),lr_img)
                txn.put('{}_hr_{}'.format(wctx.size[1],str(i).zfill(5)).encode(),hr_img)
                txn.put('{}_{}_sr_{}'.format(wctx.size[0],wctx.size[1],str(i).zfill(5)).encode(),sr_img)
        counter = wctx.increment_counter()

        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'),str(counter).encode('utf-8'))

def all_threads_inactive(threads):
    for t in threads:
        if t.is_alive():
            return False
    return True

def prepare(img_path:str,out_path:str,n_worker=0,size=[128,256],resample=Image.BICUBIC,lmdb_save=False):
    assert type(img_path)==str, 'img_path must be a string'
    assert type(out_path)==str, 'out_path must be a string'

    resize_fn = partial(resize_worker,size=size,resample=resample,lmdb_save=lmdb_save)
    files_name = [p for p in Path(img_path).rglob('*.jpg')] #rglob函数会递归地搜索指定目录下的所有文件，返回一个生成器对象，可以用于迭代,glob函数只会搜索当前目录下的文件

    if not lmdb_save:
        env = None
        path = Path(out_path)
        path.mkdir(parents=True,exist_ok=True)

        for i in ['{}_lr'.format(size[0]),f'{size[1]}_hr',f'{size[0]}_{size[1]}_sr']: 
            (path/i).mkdir(parents=True,exist_ok=True) #mkdir函数会创建指定路径的目录，如果目录已经存在，则不会报错，如果父目录不存在，则会报错，parents=True参数表示会创建父目录，exist_ok=True参数表示如果目录已经存在，则不会报错
    else:
        env = lmdb.open(out_path,map_size=1024**3*8,readahead=False) #lmdb库的创建，map_size参数表示数据库的大小，readahead参数表示是否预读数据，默认为True

    wctx = WorkingContext(resize_fn,lmdb_save,out_path,env,size)

    def process_subset(subset,wctx=wctx):
        prepare_process(subset,wctx)

    if n_worker > 1:
        img_subset = np.array_split(files_name,n_worker)
        i = 1
        with ProcessPoolExecutor(max_workers=n_worker) as executor:
            information = executor.map(process_subset,img_subset)
            for i,_ in enumerate(information):
                print('正在分批处理数据，已完成{1}/{n_worker}')
    else:
        process_subset(files_name)

    if env is not None:
        env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_path',type=str,default='./datasets/celeba_hq_256')
    parser.add_argument('--out_path',type=str,default='./datasets/celeba_lmdb')
    parser.add_argument('--n_worker',type=int,default=0)
    parser.add_argument('--size',type=tuple,default=[16,128])
    parser.add_argument('--lmdb_save',action='store_true')

    args = parser.parse_args()
    args.out_path = '{}_{}_{}'.format(args.out_path,args.size[0],args.size[1])
    prepare(args.img_path,args.out_path,args.n_worker,args.size,lmdb_save=True)
    