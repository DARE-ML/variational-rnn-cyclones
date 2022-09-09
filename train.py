from asyncore import write
import os
import io
import torch
import datetime as dt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm, trange

from torch.utils.data import DataLoader
from torch.optim import Adam
from torchmetrics import MeanSquaredError

from tensorboardX import SummaryWriter

from varnn.model import BayesRNN, BayesLSTM
from varnn.utils.markov_sampler import MarkovSamplingLoss
from cyclone_data import CycloneTracksDataset
from config import opt

matplotlib.rcParams.update({'font.size': 22})


# Train
def train(model, dataloader, epochs, writer, lr, samples):

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # MSE Metric
    metric = MeanSquaredError()

    # Number of batches
    num_batches = len(dataloader)

    # Progress bar
    pbar = trange(epochs, desc=f"Training {model.__class__.__name__}")

    # Sampler
    model.train()
    sampling_loss = MarkovSamplingLoss(model, samples=samples)

    # Random Track Id
    track_id = np.random.choice(test_dataset.track_id.detach().numpy())

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
        test_mse = evaluate(model, sampling_loss, test_dataloader)

        # Write metrics
        writer.add_scalar("train/loss", epoch_loss, epoch + 1)
        writer.add_scalar("train/mse", metric.compute(), epoch + 1)
        writer.add_scalar("test/mse", test_mse, epoch + 1)

        # Get track plot as image
        image = get_track_plot_as_image(model, sampling_loss, test_dataset, track_id=track_id)
        writer.add_image(f'Track {track_id}', image, epoch + 1)

        # Update the progress bar
        pbar.set_description(
            f"Train Loss: {epoch_loss: .4f} MSE: {metric.compute():.4f} Test MSE: {test_mse:.4f}"
        )

        # Reset metric
        metric.reset()

def evaluate(model, sampling_loss, dataloader):
    # MSE Metric
    metric = MeanSquaredError()

    # Number of batches
    num_batches = len(dataloader)

    # Progress bar
    # pbar = tqdm(, desc=f"Evaluating {model.__class__.__name__}")

    # Sampler
    model.eval()

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

def get_track_plot_as_image(model, sampling_loss, dataset, track_id):
    """Plot prediction for a track given the id"""

    # Track sequences
    X, y = dataset.get_track_data(track_id)

    # Buffer
    buf = io.BytesIO()

    # Compute prediction
    loss, out = sampling_loss(X, y, num_batches=1)
    y_hat = out.mean(0)

    # Denormalize predictions
    y = dataset.denormalize(y.detach().numpy())
    y_hat = dataset.denormalize(y_hat.detach().numpy())

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    lat, lon = y[:, 0], y[:, 1]
    lat_hat, lon_hat = y_hat[:, 0], y_hat[:, 1]
    ax.scatter(lon, lat, label="True")
    ax.scatter(lon_hat, lat_hat, label="Pred-Mean")
    plt.legend()
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Convert to Image
    image = Image.open(buf)
    image = transforms.ToTensor()(image)

    return image

if __name__ == "__main__":

    # Dataset
    ds_name = opt.ds_name
    data_dir = os.path.join(opt.root_dir, "cyclone_data")
    train_dataset = CycloneTracksDataset(
        ds_name, 
        data_dir, 
        train=True, 
        dims=opt.dims
    )
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size)

    # Test dataset
    train_min_val = train_dataset.min
    train_max_val = train_dataset.max
    test_dataset = CycloneTracksDataset(
        ds_name,
        data_dir,
        train=False,
        dims=opt.dims,
        min_=train_min_val,
        max_=train_max_val,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size)

    # Dimensions
    input_dim = len(opt.dims)
    hidden_dim = opt.hidden
    output_dim = len(opt.dims)

    # Model
    if opt.model == 'brnn':
        model = BayesRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    elif opt.model == 'blstm':
        model = BayesLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    else:
        raise(ValueError('Unknown model type: %s' % opt.model))

    # Tensorboard summary statistics
    run_name = f"{model.__class__.__name__}__{ds_name}__{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    writer = SummaryWriter(
        os.path.join(os.getcwd(), f"runs/{run_name}"),
        comment=f"{model.__class__.__name__} for {ds_name}"
    )

    train(
        model, dataloader=train_dataloader, 
        epochs=opt.epochs, 
        writer=writer, 
        lr=opt.lr,
        samples=opt.samples
    )

    writer.close()
