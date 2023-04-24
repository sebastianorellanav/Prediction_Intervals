import torch
from utils import *
from TransformerTS import TimeSeriesTransformer
from data import *
from torch.utils.data import DataLoader
from inference import *
import matplotlib.pyplot as plt

# setting device on GPU if available, else CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

enc_seq_len = 100 # length of input given to encoder
output_sequence_length = 24 # target sequence length. If hourly data and length = 48, you predict 2 days ahead
window_size = enc_seq_len + output_sequence_length # used to slice data into sub-sequences
step_size = 1 # Step size, i.e. how many time steps does the moving window move at each step
data = load_data().reshape(-1,1)
data = torch.Tensor(data).to(device)
# Remove test data from dataset
training_data = data[:-(round(len(data)*0.2))]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
# Should be training data indices only
training_indices = get_indices_entire_sequence(
    data=training_data, 
    window_size=window_size, 
    step_size=step_size)

# Making instance of custom dataset class
training_data = TransformerDataset(
    data=torch.tensor(training_data).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    target_seq_len=output_sequence_length
    )

training_dataloader = DataLoader(training_data)

test_data = data[(round(len(data)*0.8)):]

# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc. 
# Should be training data indices only
training_indices = get_indices_entire_sequence(
    data=test_data, 
    window_size=window_size, 
    step_size=step_size)

# Making instance of custom dataset class
test_dataset = TransformerDataset(
    data=torch.tensor(test_data).float(),
    indices=training_indices,
    enc_seq_len=enc_seq_len,
    target_seq_len=output_sequence_length
    )
test_dataloader = DataLoader(test_dataset, 64)

model = TimeSeriesTransformer(
    input_size=1,
    batch_first=True,
    num_predicted_features=1
    )
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

criterion = torch.nn.MSELoss()

# Generate masks
tgt_mask = generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=output_sequence_length
    ).to(device)

src_mask = generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=enc_seq_len
    ).to(device)

# Iterate over all epochs
for epoch in range(100):
    print(f"epoch: {epoch}")
    # Iterate over all (x,y) pairs in training dataloader
    for i, (src, tgt, tgt_y) in enumerate(training_data):

        # zero the parameter gradients
        optimizer.zero_grad()
        # Make forecasts
        prediction = model(src, tgt, src_mask, tgt_mask)
        #print(tgt_y.shape)
        #print(prediction.squeeze(-1).shape)
        # Compute and backprop loss
        loss = criterion(tgt_y, prediction.squeeze(-1))
        
        loss.backward()

        # Take optimizer step
        optimizer.step()

# Iterate over all (x,y) pairs in validation dataloader
model.eval()
predictions = torch.tensor([])
with torch.no_grad():

    for i, (src, _, tgt_y) in enumerate(test_dataloader):

        pred = predict(
            model=model, 
            src=src, 
            forecast_window=output_sequence_length,
            batch_size=src.shape[0]
            ).squeeze(-1)
        predictions = torch.cat(predictions, pred)
        #loss = criterion(tgt_y, prediction)

    test_dataloader = DataLoader(test_dataset, test_dataset.__len__())
    _, _, tgt_y = next(iter(test_dataloader))
    loss = criterion(tgt_y, prediction)
    print(loss)

    plt.figure()
    plt.plot(tgt_y.cpu().numpy(), label="target")
    plt.plot(pred.cpu().detach().numpy(), label="prediction")
    plt.legend()

    plt.figure()
    plt.plot(tgt_y.cpu().numpy()[:50], label="target")
    plt.plot(pred.cpu().detach().numpy()[:50], label="prediction")
    plt.legend()
    plt.show()

