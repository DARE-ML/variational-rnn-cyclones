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
dims = (1, 2)
root_dir = os.path.join(os.getcwd(), "cyclone_data")
train_dataset = CycloneTracksDataset(
                                    ds_name, root_dir, 
                                    train=True,
                                    dims=dims
                                )
train_dataloader = DataLoader(train_dataset, batch_size=1024)


# Test dataset
train_min_val = train_dataset.min
train_max_val = train_dataset.max
test_dataset = CycloneTracksDataset(
                                    ds_name, root_dir, 
                                    train=False, 
                                    dims=dims,
                                    min_=train_min_val, 
                                    max_=train_max_val
                                )
test_dataloader = DataLoader(test_dataset, batch_size=1024)

# Dimensions
input_dim = len(dims)
hidden_dim = 16
output_dim = len(dims)

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
            loss.backward(retain_graph=True)
            optimizer.step()

            # Update metric
            metric.update(outputs.mean(dim=0), labels)

        pbar.set_description(f"MSE: {metric.compute():.4f}")


def evaluate(model, dataloader):
    # MSE Metric
    metric = MeanSquaredError()

    # Number of batches
    num_batches = len(dataloader)

    # Progress bar
    pbar = tqdm(enumerate(dataloader), desc=f"Training {model.__class__.__name__}")
    
    # Sampler
    model.eval()
    sampling_loss = MarkovSamplingLoss(brnn)

    for idx, (seq, labels, tracks) in pbar:
        
        # Compute sampling loss
        outputs = sampling_loss(seq, labels, num_batches, testing=True)

        # Update metric
        metric.update(outputs.mean(dim=0), labels)

        pbar.set_description(f"MSE: {metric.compute():.4f}")

    
    



train(brnn, dataloader=train_dataloader, epochs=50)

evaluate(brnn, dataloader=test_dataloader)

