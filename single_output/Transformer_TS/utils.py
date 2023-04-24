import pandas as pd
import torch


def get_indices_entire_sequence(data: pd.DataFrame, window_size: int, step_size: int) -> list:
        stop_position = len(data)-1 # 1- because of 0 indexing
        
        # Start the first sub-sequence at index position 0
        subseq_first_idx = 0
        
        subseq_last_idx = window_size
        
        indices = []
        
        while subseq_last_idx <= stop_position:

            indices.append((subseq_first_idx, subseq_last_idx))
            
            subseq_first_idx += step_size
            
            subseq_last_idx += step_size

        return indices

def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)