import random
import collections
from operator import itemgetter
from dataset import Token
import torch
from torch.nn.utils.rnn import pad_sequence

class Collator:
    def __init__(self, len_data, batch_size, size_gap = 5):
        self.len_data = len_data
        self.size_gap = size_gap
        self.batch_size = batch_size
        self.data_size = len(len_data)
        
    def sample(self) :
        batch_map = collections.defaultdict(list)
        idx_list = []
        batch_index = []
    
        for idx in range(self.data_size) :
            src_idx, tar_idx = self.len_data[idx]
            
            src_group = src_idx // self.size_gap
            tar_group = tar_idx // self.size_gap
            
            batch_map[src_group, tar_group].append(idx)
            
        batch_key = list(batch_map.keys())
        batch_key = sorted(batch_key, key=itemgetter(0,1), reverse=True) 
        # sorting idx list based on size group
        for key in batch_key :
            idx_list.extend(batch_map[key])
    
        # slicing batch_size
        for i in range(0, self.data_size, self.batch_size) :
            batch_index.append(idx_list[i:i+self.batch_size])
    
        random.shuffle(batch_index)
        return batch_index
    
    def __call__(self, batch_samples):   
        src_tensor = []
        tar_tensor = []
        for src_idx, tar_idx in batch_samples:
            src_tensor.append(torch.tensor(src_idx))
            tar_tensor.append(torch.tensor(tar_idx + [Token.PAD]))
        
        src_tensor = pad_sequence(src_tensor, batch_first=True, padding_value=Token.PAD)
        tar_tensor = pad_sequence(tar_tensor, batch_first=True, padding_value=Token.PAD)

        return {'encoder_in' : src_tensor, 
                'decoder_in' : tar_tensor[:,:-1], 
                'decoder_out' : tar_tensor[:,1:]}
