import torch
import torch.nn as nn

# from trainTFCtask2 import train_op
import pdb

""" 
    用于统计class数和每个class的样本占比
"""

src_file = ''

label = torch.load(src_file)
n_sample = label.shape
