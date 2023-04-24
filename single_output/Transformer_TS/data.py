import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset

"""
load_data
"""
def load_data(path = None, days=1000):
    if path == None: #return temperature data
        temperature_data = np.zeros(24*days)
        for i in range(days):
            # Generate daily temperature pattern
            daily_temp = np.concatenate([
                np.random.normal(loc=5, scale=1, size=3),  # Morning
                np.random.normal(loc=10, scale=1, size=4),  # Morning
                np.random.normal(loc=20, scale=1, size=10), # Afternoon
                np.random.normal(loc=10, scale=1, size=4),   # Night
                np.random.normal(loc=5, scale=1, size=3),  # Morning
            ])
            temperature_data[i*24:(i+1)*24] = daily_temp
        return temperature_data
    else:
        pass # return data from path

"""
Dataset
"""
class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.
    
    """
    def __init__(self, data: torch.tensor,indices: list, enc_seq_len: int, target_seq_len: int) -> None:

        super().__init__()
        self.indices = indices
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.target_seq_len = target_seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Get the first element of the i'th tuple in the list self.indicesasdfas
        start_idx = self.indices[index][0]

        # Get the second (and last) element of the i'th tuple in the list self.indices
        end_idx = self.indices[index][1]

        sequence = self.data[start_idx:end_idx]

        #print("From __getitem__: sequence length = {}".format(len(sequence)))

        src, trg, trg_y = self.get_src_trg(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            target_seq_len=self.target_seq_len
            )

        return src, trg, trg_y
    
    def get_src_trg(self, sequence: torch.Tensor, enc_seq_len: int, target_seq_len: int) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:


        assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"
        
        # encoder input
        src = sequence[:enc_seq_len] 
        
        # decoder input. As per the paper, it must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = sequence[enc_seq_len-1:len(sequence)-1]
        
        assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

        # The target sequence against which the model output will be compared to compute loss
        trg_y = sequence[-target_seq_len:]

        assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y.squeeze(-1) # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len] 
