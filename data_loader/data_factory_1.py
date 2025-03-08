from data_loader.dataloader_1 import Wind_Dataset
from torch.utils.data import DataLoader

def _get_data(config,flag,result_df):

    timeenc = 0 if config.embed != 'timeF' else 1

    if flag == 'train':
        shuffle_flag = True
        drop_last = True
        batch_size = config.batch_size
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = config.batch_size

    data_set = Wind_Dataset(
        root_path=config.root_path,
        data_S_path=config.data_S_path,
        data_G_path=config.data_G_path,
        flag=flag,
        hier_df=result_df,
        size=[config.seq_len, config.label_len,config.pred_len],
        scale=True,
        data_split=config.data_split,
        timeenc=timeenc,
        freq = config.freq,
        site = config.site
    )

    print(flag, len(data_set),"data_x",data_set.data_x.shape,"data_y",data_set.data_y.shape,"data_time",data_set.data_stamp.shape)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)

    return data_set, data_loader