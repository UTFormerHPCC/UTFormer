import torch
from torch.utils.data import Dataset, DataLoader

# 定义一个自定义的数据集类
class TrafficDataset(Dataset):
    def __init__(self, data_file, label_file):
        # 加载数据和标签
        self.data = torch.load(data_file)
        self.labels = torch.load(label_file)

    def __len__(self):
        # 返回数据集中的总样本数
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引返回一个样本及其标签
        return self.data[idx], self.labels[idx]

