from torch._C import get_autocast_gpu_dtype
import torch.nn as nn
import torch
from utils import *
def predict(model: nn.Module, src: torch.Tensor, forecast_window: int,batch_size: int,device,batch_first: bool=True) -> torch.Tensor:

    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 1

    # Take the last value of thetarget variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[:, -1, 0] # shape [1, batch_size, 1]
    print(tgt.shape)

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_size == 1 and batch_first == True:
        tgt = tgt.unsqueeze(1).unsqueeze(1) # change from [1] to [1, 1, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_first == True and batch_size > 1:
        tgt = tgt.unsqueeze(-1).unsqueeze(-1)
        print(tgt.shape)

  
    
    # Iteratively concatenate tgt with the first element in the prediction
    for i in range(forecast_window-1):
        print(f"iteration: {i}")
        dim_a = tgt.shape[1] 

        dim_b = src.shape[1] 

        tgt_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a,
            ).to(device)

        src_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b
            ).to(device) 
        # Make prediction
        prediction = model(src, tgt, src_mask, tgt_mask) 

        # If statement simply makes sure that the predicted value is 
        # extracted and reshaped correctly
        if batch_first == False:

            # Obtain the predicted value at t+1 where t is the last time step 
            # represented in tgt
            last_predicted_value = prediction[-1, :, :] 

            # Reshape from [batch_size, 1] --> [1, batch_size, 1]
            last_predicted_value = last_predicted_value.unsqueeze(0)

        else:

            # Obtain predicted value
            last_predicted_value = prediction[:, -1, :]

            # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
            last_predicted_value = last_predicted_value.unsqueeze(-1)

        # Detach the predicted element from the graph and concatenate with 
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)
        src_mask.cpu()
        tgt_mask.cpu()

    dim_a = tgt.shape[1] 

    dim_b = src.shape[1] 

    tgt_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a,
        ).to(device)

    src_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b
        ).to(device) 

    # Make final prediction
    final_prediction = model(src, tgt, src_mask, tgt_mask)
    src_mask.cpu()
    tgt_mask.cpu()
    return final_prediction
