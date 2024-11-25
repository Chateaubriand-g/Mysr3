

def create_dataset(dataset_opt,phase):
    """phase is train or test"""
    mode = dataset_opt['mode']
    from class_dataset import Dataset
    dataset = Dataset(dataroot=dataset_opt['dataroot'],
                      datatype=dataset_opt['datatype'],
                      lr=dataset_opt['l_resolution'],
                      hr=dataset_opt['h_resolution'],
                      split=phase,
                      need_lr=(mode=='LRHR'),
                      datalen=dataset_opt['data_len'])
    return dataset

def create_dataloader(dataset,dataset_opt,phase):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,
                            batch_size=dataset_opt['batch_size'],
                            shuffle=dataset_opt['shuffle'],
                            num_workers=dataset_opt['num_workers'],
                            pin_memory=True)  #pin_memory=True表示数据加载到GPU时使用锁页内存，可以加快数据传输速度，锁页内存是CPU和GPU都可以访问的内存
    return dataloader