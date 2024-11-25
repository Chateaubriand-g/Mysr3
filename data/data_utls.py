import torch
import os
import random
from torchvision import transforms


IMG_EXTENSIONS = ['jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif',
                   'tiff', 'webp', 'JPEG', 'JPG', 'PNG']


#判断输入是否是合理的图片文件
def is_imagefile(filename):
    return any(filename.endswith(extensions) for extensions in IMG_EXTENSIONS)

#获取指定路径下的所有图片文件，并按文件名排序,为了保证不同次运行结果一致
def get_sorted_paths_from_images(path):
    assert os.path.isdir(path), '{}不是合法的路径'.format(path)
    image_path = []
    for root,_,file_name in sorted(os.walk(path)):  # os.walk()返回一个三元组(root str,dirs list,files list) root是当前目录的路径，
                                                    # dirs是当前路径下所有子目录，files是当前路径下所有非目录文件
        for name in sorted(file_name):
            if is_imagefile(name):
                image_path.append(os.path.join(root,name))
    assert image_path, '没有找到图片文件'  # image_path为list，当list为空时，返回false
    return sorted(image_path)

#获取指定路径下的所有图片文件，但并不排序
def get_paths_from_images(path):
    assert os.path.isdir(path), '{}不是合法的路径'.format(path)
    image_path = []
    for root,_,file_name in os.walk(path):
        for name in file_name:
            if is_imagefile(name):
                image_path.append(os.path.join(root,name))
    assert image_path, '没有找到图片文件'
    return sorted(image_path)

#进行图片增强操作
# def augment(img_list,hflip=True,vflip=True,rotate=True,mode='val'):
#     hflip = hflip and (mode == 'train' and random.random() < 0.5)
#     vflip = vflip and (mode == 'train' and random.random() < 0.5)
#     rotate = rotate and (mode == 'train' and random.random() < 0.5)

#     def augment_inner(img):
#         if hflip: img = img[:, ::-1, :]
#         if vflip: img = img[::-1, :, :]
#         if rotate: img = img.transpose(1, 0, 2)
#         return img
    
#     return [augment_inner(img) for img in img_list]
def augment(img_list,model='val'):
    if model == 'train':
        imgs = torch.stack(img_list,dim=0)
        imgs = transforms.RandomHorizontalFlip()(imgs)
        imgs = torch.unstack(imgs,dim=0)
    img_list = imgs
    return img_list

PIL_toTensor = transforms.to_tensor

Tensor_toPIL = transforms.Compose(
    transforms.Lambda(lambda x:x*255),
    transforms.Lambda(lambda x:x.dtype(torch.uint8)),
    transforms.ToPILImage()
)