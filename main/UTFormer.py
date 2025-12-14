import math
import torch
import torch.nn as nn
from learning_op import train_op, evaluate_op
import pdb
from mtt_1 import MTT_dropping
from traffic_loader import TrafficDataset
from torch.utils.data import DataLoader

def buildup_model(
        seq_len=1500, 
        embed_n=90, 
        embed_d=16, 
        trans_h=5, 
        trans_d1=256,
        n_block=4,
        keep_rate=1.0,
        n_class=2):

    model = MTT_dropping(seq_len, embed_n, embed_d, trans_h, trans_d1, n_block, keep_rate, n_class)
    
    return model


def main(pretrain=False, eval_only=False):

    """ 
        num_class:
            cross-platform-android  119
    """


    seq_len = 1500
    byte_per_token = 16
    embed_n = 1440 // byte_per_token
    embed_d = 64
    trans_h = 4
    trans_d1 = embed_d*4
    n_block = 4
    keep_rate = 0.8
    # n_class = 125 # android
    # n_class = 121 # android
    n_class = 155 # ios

    batch_size = 512
    n_epoch = 28

    model = buildup_model(
        seq_len=seq_len, 
        embed_n=embed_n, 
        embed_d=embed_d, 
        trans_h=trans_h, 
        trans_d1=trans_d1,
        n_block=n_block,
        keep_rate=keep_rate,
        n_class=n_class)
    
    if (pretrain == False and eval_only == False):
        train_op(model=model,
                n_class=n_class,
                batch_size=batch_size,
                n_epochs=n_epoch)
    
    if (pretrain == True and eval_only == False):
        model.load_state_dict(torch.load('./best/best_bak.pt'))
        train_op(model=model,
                n_class=n_class,
                batch_size=batch_size,
                n_epochs=n_epoch)
        
    if (pretrain == True and eval_only == True):
        # model.load_state_dict(torch.load('./best/best_andr_80.pt'))
        # test_data_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/android/test_data.pt'
        # test_label_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/android/test_label.pt
        
        model.load_state_dict(torch.load('./best/best_ios_80.pt'))
        test_data_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/test_data.pt'
        test_label_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/test_label.pt'

        val_dataset = TrafficDataset(test_data_path, test_label_path)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        model = model.to('cuda:0')
        
        evaluate_op(
            model=model,
            n_class=n_class,
            data_loader=val_data_loader
        )
    
    if (pretrain == False and eval_only == True):

        test_data_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/android/test_data.pt'
        test_label_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/android/test_label.pt'
        val_dataset = TrafficDataset(test_data_path, test_label_path)
        val_data_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        if not torch.cuda.is_available():
            print('Fail to use GPU')
            device = 'cpu'
        else:
            device = 'cuda:0'
            
        model = model.to(device)
        
        evaluate_op(
            model=model,
            n_class=n_class,
            data_loader=val_data_loader
        )





if __name__ == '__main__':
    # main(pretrain=False, eval_only=False)
    main(pretrain=True, eval_only=True)

