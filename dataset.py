import torch
from torch.utils.data import Dataset, Subset
import collections
import random
from enum import IntEnum

class Token(IntEnum) :
    PAD = 0
    UNK = 1
    SOS = 2
    EOS = 3

class TranslationDataset(Dataset) :
    def __init__(self, en_index, de_index, situation_data, max_size, val_ratio=0.1) :
        super(TranslationDataset , self).__init__()
        self.max_size = max_size
        self.val_ratio = val_ratio
        self.situation_data = situation_data
        self.idx_data = self.build_data(en_index, de_index)
        
    def __len__(self) :
        return len(self.idx_data)

    def __getitem__(self , idx) :
        return self.idx_data[idx]

    def build_data(self, en_data, de_data) :
        data_len = len(en_data)
        idx_data = [(en_data[i][-self.max_size:],de_data[i][-self.max_size:]) for i in range(data_len)]
        return idx_data
    
    def split(self):
        data_size = len(self)
        index_map = collections.defaultdict(list)
        for idx in range(data_size):
            label = self.situation_data[idx]
            index_map[label].append(idx)

        train_data = []
        val_data = []

        label_size = len(index_map)
        for label in index_map.keys():
            idx_list = index_map[label]
            sample_size = int(len(idx_list) * self.val_ratio)

            val_index = random.sample(idx_list, sample_size)
            train_index = list(set(idx_list) - set(val_index))

            train_data.extend(train_index)
            val_data.extend(val_index)

        random.shuffle(train_data)
        random.shuffle(val_data)

        train_dset = Subset(self, train_data)
        val_dset = Subset(self, val_data)
        return train_dset, val_dset
