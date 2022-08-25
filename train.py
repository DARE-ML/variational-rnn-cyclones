import os
import torch

from torch.utils.data import DataLoader

from varnn.model.bayes_rnn import BayesRNN
from varnn.utils.markov_sampler import MarkovSampler

from cyclone_data import CycloneTracksDataset


# Dataset
ds_name = 'north_indian_ocean'
root_dir = os.path.join(os.getcwd(), 'cyclone_data')
dataset = CycloneTracksDataset(ds_name, root_dir, train=True)
dataloader = DataLoader(dataset, batch_size=1024)

# Dimensions
input_dim = 3
hidden_dim = 16
output_dim = 3

# Model
brnn = BayesRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
sampler = MarkovSampler(brnn)

for X, y in dataloader:
    # During Training
    loss, mse_loss, outputs = sampler(X, y, num_batches=1, testing=False)
    print(loss)
    print(mse_loss)
    print(outputs.shape)
