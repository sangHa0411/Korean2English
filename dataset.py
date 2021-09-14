import torch
from torch.utils.data import Dataset, Subset, random_split

class TranslationDataset(Dataset) :
    def __init__(self, en_index, de_index, max_size, val_ratio=0.1) :
        super(TranslationDataset , self).__init__()
        self.max_size = max_size
        self.val_ratio = val_ratio
        self.idx_data = self.build_data(en_index, de_index)
        
    def __len__(self) :
        return len(self.idx_data)

    def __getitem__(self , idx) :
        return self.idx_data[idx]

    def build_data(self, en_data, de_data) :
        data_len = len(en_data)
        idx_data = [(en_data[i][-self.max_size:],de_data[i][-self.max_size:]) for i in range(data_len)]
        return idx_data
    
    def split(self) :
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        
        return train_set, val_set