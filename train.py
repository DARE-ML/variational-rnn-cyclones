import os
from pickletools import optimize
import torch

from tqdm import tqdm, trange

from torch.utils.data import DataLoader
from torch.optim import Adam

from torchmetrics import MeanSquaredError

from varnn.model.bayes_rnn import BayesRNN
from varnn.utils.markov_sampler import MarkovSamplingLoss
from cyclone_data import CycloneTracksDataset


# Dataset
ds_name = "north_indian_ocean"
root_dir = os.path.join(os.getcwd(), "cyclone_data")
dataset = CycloneTracksDataset(ds_name, root_dir, train=True)
dataloader = DataLoader(dataset, batch_size=1024)

# Dimensions
input_dim = 3
hidden_dim = 16
output_dim = 3

# Model
brnn = BayesRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)


# Train
def train(model, dataloader, epochs):
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=0.1)

    # MSE Metric
    metric = MeanSquaredError()

    # Number of batches
    num_batches = len(dataloader)

    # Progress bar
    pbar = trange(epochs, desc=f"Training {model.__class__.__name__}")
    
    # Sampler
    model.train()
    sampling_loss = MarkovSamplingLoss(brnn)

    for epoch in pbar:
        
        for seq, labels, tracks in dataloader:
            
            # Reset the gradients
            model.zero_grad()
            
            # Compute sampling loss
            loss, outputs = sampling_loss(seq, labels, num_batches)
            
            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update metric
            metric.update(outputs.mean(dim=0), labels)

        pbar.set_description(f"MSE: {metric.compute():.4f}")



train(brnn, dataloader=dataloader, epochs=50)