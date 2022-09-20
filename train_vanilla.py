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

def train(models, dataloader, epochs, writer, lr, samples):

    # MSE Metric
    metric = MeanSquaredError()
    mse_loss = nn.MSELoss()

    # MSE
    train_rmse_list = torch.zeros(samples)
    test_rmse_list = torch.zeros(samples)

    for sample in range(0, samples):

        # Model to train
        model = models[sample]
        
        # Optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        print(f"Model: {model.__class__.__name__} Sample: {sample}")

        # Sampler
        model.init_weights()
        model.train()

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
            test_rmse = evaluate(model, test_dataloader)
            train_rmse = torch.sqrt(metric.compute())

            # Update the progress bar
            pbar.set_description(
                f"Train Loss: {epoch_loss: .4f} RMSE: {train_rmse:.4f} Test RMSE: {test_rmse:.4f}"
            )

        # Append MSE
        train_rmse_list[sample] = train_rmse
        test_rmse_list[sample] = test_rmse

        # Update model
        models[sample] = model

    # RMSE
    train_rmse_mean = train_rmse_list.mean().detach().numpy()
    train_rmse_std = train_rmse_list.std().detach().numpy()
    test_rmse_mean = test_rmse_list.mean().detach().numpy()
    test_rmse_std = test_rmse_list.std().detach().numpy()

    return models, float(train_rmse_mean), float(train_rmse_std), float(test_rmse_mean), float(test_rmse_std)

def evaluate(model, dataloader):

    model.eval()

    # MSE Metric
    metric = MeanSquaredError()

    for idx, (seq, labels, tracks) in enumerate(dataloader):

        # Output
        out = model(seq)

        # Update metric
        metric.update(out, labels)

    # Compute RMSE
    rmse = torch.sqrt(metric.compute())

    return rmse


def evaluate_energy_score(models, dataloader):
    
    # Number of models in ensemble
    samples = len(models)

    # List to store outputs
    ensemble_out_list = []
    label_list = []

    for sample, model in enumerate(models):

        # Sampler
        model.eval()

        out_list = []

        for batch_idx, (seq, labels, tracks) in enumerate(dataloader):

            # Predict labels
            out = model(seq)

            # Add to list
            out_list.append(out)

            if sample == 0:
                label_list.append(labels)

        # Append sample outputs to ensemble
        sample_outputs = torch.concat(out_list)
        ensemble_out_list.append(sample_outputs)
                
    # Concat the outputs from the ensemble
    ensemble_out_list = torch.stack(ensemble_out_list)
    label_list = torch.concat(label_list)

    # Energy scores per sample
    es_list = []

    # Evaluate energy score
    for seq_idx in range(len(label_list)):

        # observations
        obs = label_list[seq_idx].detach().numpy().tolist()

        # Samples
        fc_sample = ensemble_out_list[:, seq_idx, :].detach().numpy().reshape(len(obs), samples).ravel().tolist()

        # Convert to rtypes
        obs = FloatArray(obs)
        fc_sample = rbase.matrix(FloatArray(fc_sample), nrow=len(obs), ncol=samples)

        # Energy scores
        es_list.append(np.asarray(scoringRules.es_sample(y=obs, dat=fc_sample)))
    
    return np.mean(es_list)


def remove_outliers(x):
    """Remove outlier samples"""
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    up = q3 + 1.5 * iqr
    low = q1 - 1.5 * iqr
    return x[(x > low) & (x < up)]


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
        models = [
            VanillaRNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            for s in range(opt.samples)
        ]
    elif opt.model == 'lstm':
        models = [
            VanillaLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )
            for s in range(opt.samples)
        ]
    else:
        raise(ValueError('Unknown model type: %s' % opt.model))

    # Tensorboard summary statistics
    run_name = f"{models[0].__class__.__name__}__{ds_name}__{opt.features}__{dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    run_path = os.path.join(os.getcwd(), f"runs/{run_name}")
    writer = SummaryWriter(
        run_path,
        comment=f"{models[0].__class__.__name__} for {opt.features} features in {ds_name}"
    )

    res = train(
        models, dataloader=train_dataloader, 
        epochs=opt.epochs, 
        writer=writer, 
        lr=opt.lr,
        samples=opt.samples
    )

    models, train_rmse_mean, train_rmse_std, test_rmse_mean, test_rmse_std = res

    # ## Remove outliers
    # train_mse = remove_outliers(train_rmse)
    # test_mse = remove_outliers(test_rmse)

    print(f"Train RMSE: {train_rmse_mean:.6f} + {train_rmse_std:.6f}")
    print(f"Test RMSE: {test_rmse_mean:.6f} + {test_rmse_std:.6f}")

    # Evaluate test sample energy score
    train_es = evaluate_energy_score(
        models,
        train_dataloader
    )

    # Test mean energy score
    print(f"Train Energy Score: {train_es:.4f}")

    # Evaluate test sample energy score
    test_es = evaluate_energy_score(
        models,
        test_dataloader
    )

    # Test mean energy score
    print(f"Mean Energy Score: {test_es:.4f}")

    # Save results to path
    results = [
        {
            'model': models[0].__class__.__name__,
            'ds_name': ds_name,
            'run_name': run_name,
            'train_rmse': {
                'mean': train_rmse_mean,
                'std': train_rmse_std
            },
            'test_rmse': {
                'mean': test_rmse_mean,
                'std': test_rmse_std
            },
            'train_es': train_es,
            'test_es': test_es
        }
    ]

    result_file = os.path.join(run_path, 'results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
        

    writer.close()
