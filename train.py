import os
import torch
import datetime as dt
import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm, trange

from torch.utils.data import DataLoader
from torch.optim import Adam

from torchmetrics import MeanSquaredError

from tensorboardX import SummaryWriter

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

# Tensorboard summary statistics
run_name = f"{brnn.__class__.__name__}__{ds_name}__{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
writer = SummaryWriter(
                        os.path.join(os.getcwd(), f"runs/{run_name}"), 
                        comment=f"{brnn.__class__.__name__} for {ds_name}"
                    )


# Train
def train(model, dataloader, epochs):
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=0.001)

    # MSE Metric
    metric = MeanSquaredError()

    # Number of batches
    num_batches = len(dataloader)

    # Progress bar
    pbar = trange(epochs, desc=f"Training {model.__class__.__name__}")
    
    # Sampler
    model.train()
    sampling_loss = MarkovSamplingLoss(model)

    for epoch in pbar:

        epoch_loss = 0
        
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

            # Total loss
            epoch_loss += loss.detach().numpy()

        # Validation
        test_mse = evaluate(model, test_dataloader)

        # Write metrics
        writer.add_scalar('train/loss', epoch_loss, epoch+1)
        writer.add_scalar('train/mse', metric.compute(), epoch+1)
        writer.add_scalar('test/mse', test_mse, epoch+1)

        # Update the progress bar
        pbar.set_description(f"Train Loss: {epoch_loss: .4f} MSE: {metric.compute():.4f} Test MSE: {test_mse:.4f}")

        # Reset metric
        metric.reset()


def evaluate(model, dataloader):
    # MSE Metric
    metric = MeanSquaredError()

    # Number of batches
    num_batches = len(dataloader)

    # Progress bar
    # pbar = tqdm(, desc=f"Evaluating {model.__class__.__name__}")
    
    # Sampler
    model.eval()
    sampling_loss = MarkovSamplingLoss(model)

    for idx, (seq, labels, tracks) in enumerate(dataloader):
        
        # Compute sampling loss
        outputs = sampling_loss(seq, labels, num_batches, testing=True)

        # Update metric
        metric.update(outputs.mean(dim=0), labels)

    # Compute mse
    mse = metric.compute()

    # Reset metric
    metric.reset()

    return mse
    


def plot_track_prediction(model, dataset, track_id):
    """Plot prediction for a track given the id"""

    # Track sequences
    X, y = dataset.get_track_data(track_id)

    # Compute prediction
    sampling_loss = MarkovSamplingLoss(model)
    loss, out = sampling_loss(X, y, num_batches=1)
    y_hat = out.mean(0)

    # Denormalize predictions
    y = dataset.denormalize(y.detach().numpy())
    y_hat = dataset.denormalize(y_hat.detach().numpy())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    lat, lon = y[:, 0], y[:, 1]
    lat_hat, lon_hat = y_hat[:, 0], y_hat[:, 1]
    ax.scatter(lon, lat, label='True')
    ax.scatter(lon_hat, lat_hat, label='Pred-Mean')
    plt.legend()
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    plt.savefig('plot.png')
    




train(brnn, dataloader=train_dataloader, epochs=150)

track_id = np.random.choice(test_dataset.track_id.detach().numpy())
plot_track_prediction(brnn, test_dataset, track_id=track_id)

writer.close()