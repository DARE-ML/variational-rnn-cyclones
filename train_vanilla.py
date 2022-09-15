import os
import io
import time
import json
import torch
import datetime as dt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm, trange

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchmetrics import MeanSquaredError

from tensorboardX import SummaryWriter

# Using R inside python
import rpy2
from rpy2.robjects.vectors import FloatArray
from rpy2.robjects.packages import importr
rbase = importr('base')
scoringRules = importr('scoringRules')

from varnn.model import (VanillaRNN,
                        VanillaLSTM)
from varnn.utils.markov_sampler import MarkovSamplingLoss
from cyclone_data import CycloneTracksDataset
from config import opt

matplotlib.rcParams.update({'font.size': 22})

def train(model, dataloader, epochs, writer, lr, samples):

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # MSE Metric
    metric = MeanSquaredError()
    mse_loss = nn.MSELoss()

    # Random Track Id
    track_id = np.random.choice(test_dataset.track_id.detach().numpy())

    for sample in range(1, samples+1):

        print(f"Model: {model.__class__.__name__} Sample: {sample}")

        # Sampler
        model.init_weights()
        model.train()

        # MSE
        train_mse_list = torch.zeros(samples)
        test_mse_list = torch.zeros(samples)

        # Progress bar
        pbar = trange(epochs, desc=f"Training {model.__class__.__name__}")

        for epoch in pbar:

            # Reset metric
            metric.reset()
            epoch_loss = 0

            for seq, labels, tracks in dataloader:

                # Reset the gradients
                model.zero_grad()

                # Predict and compute loss
                output = model(seq)
                loss = mse_loss(output, labels)

                # Backpropagation
                loss.backward()
                optimizer.step()

                # Update metric
                metric.update(output, labels)

                # Total loss
                epoch_loss += loss.detach().numpy()

            # Validation
            test_mse = evaluate(model, test_dataloader)

            # Write metrics
            # writer.add_scalar("train/loss", epoch_loss, epoch + 1)
            # writer.add_scalar("train/mse", metric.compute(), epoch + 1)
            # writer.add_scalar("test/mse", test_mse, epoch + 1)


            # if opt.features in ('location', 'both'):
            #     # Get track plot as image
            #     image = get_track_plot_as_image(model, sampling_loss, test_dataset, track_id=track_id)
            #     writer.add_image(f'Track {track_id}', image, epoch + 1)

            # Update the progress bar
            pbar.set_description(
                f"Train Loss: {epoch_loss: .4f} MSE: {metric.compute():.4f} Test MSE: {test_mse:.4f}"
            )

            # Append MSE
            train_mse_list[sample-1] = metric.compute()
            test_mse_list[sample-1] = test_mse
        
    return train_mse_list, test_mse_list

def evaluate(model, dataloader):

    model.eval()

    # MSE Metric
    metric = MeanSquaredError()

    for idx, (seq, labels, tracks) in enumerate(dataloader):

        # Output
        outputs = model(seq)

        # Update metric
        metric.update(outputs, labels)

    return metric.compute()

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
    plt.close()
    buf.seek(0)

    # Convert to Image
    image = Image.open(buf)
    image = transforms.ToTensor()(image)

    return image

def evaluate_energy_score(model, samples, dataloader):

    # Number of batches
    num_batches = len(dataloader)
    
    # Sampler
    model.eval()
    sampling_loss = MarkovSamplingLoss(model, samples=samples)

    energy_scores = []

    for batch_idx, (seq, labels, tracks) in enumerate(dataloader):

        # Compute sampling loss
        outputs = sampling_loss(seq, labels, num_batches, testing=True)

        # Evaluate energy score
        for seq_idx in range(len(labels)):

            # observations
            obs = labels[seq_idx].detach().numpy().tolist()

            # Samples
            fc_sample = outputs[:, seq_idx, :].detach().numpy().reshape(len(obs), samples).ravel().tolist()

            # Convert to rtypes
            obs = FloatArray(obs)
            fc_sample = rbase.matrix(FloatArray(fc_sample), nrow=len(obs), ncol=samples)

            # Energy scores
            energy_scores.append(np.asarray(scoringRules.es_sample(y=obs, dat=fc_sample)))
    
    return energy_scores


if __name__ == "__main__":

    torch.manual_seed(int(time.time()))

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
    if opt.model == 'rnn':
        model = VanillaRNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    if opt.model == 'lstm':
        model = VanillaLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
    else:
        raise(ValueError('Unknown model type: %s' % opt.model))

    # Tensorboard summary statistics
    run_name = f"{model.__class__.__name__}__{ds_name}__{opt.features}__{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    run_path = os.path.join(os.getcwd(), f"runs/{run_name}")
    writer = SummaryWriter(
        run_path,
        comment=f"{model.__class__.__name__} for {opt.features} features in {ds_name}"
    )

    train_mse, test_mse = train(
        model, dataloader=train_dataloader, 
        epochs=opt.epochs, 
        writer=writer, 
        lr=opt.lr,
        samples=opt.samples
    )

    print(f"Train MSE: {train_mse.mean():.6f} + {train_mse.std():.6f}")
    print(f"Test MSE: {test_mse.mean():.6f} + {test_mse.std():.6f}")
    

    # sampling_loss = MarkovSamplingLoss(model, opt.samples)

    # train_mse_mean, train_mse_std = evaluate(
    #     model=model,
    #     sampling_loss=sampling_loss,
    #     dataloader=train_dataloader
    # )

    # # Evaluate test sample energy score
    # es_list = evaluate_energy_score(
    #     model,
    #     opt.samples,
    #     test_dataloader
    # )

    # # Test mean energy score
    # es_mean = np.concatenate(es_list).mean()
    # print(f"Mean Energy Score: {es_mean:.4f}")

    # # Save results to path
    # results = [
    #     {
    #         'model': model.__class__.__name__,
    #         'ds_name': ds_name,
    #         'run_name': run_name,
    #         'train_mse': {
    #             "mean": train_mse_mean,
    #             "std": train_mse_std
    #         },
    #         'test_mse': {
    #             "mean": test_mse_mean, 
    #             "std": test_mse_std
    #         },
    #         'energy_score': es_mean
    #     }
    # ]

    # result_file = os.path.join(run_path, 'results.json')
    # with open(result_file, 'w') as f:
    #     json.dump(results, f, indent=4)
        

    writer.close()
