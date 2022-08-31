import os
import yaml

import torch

import numpy as np
import scipy.io as sio

from torch.utils.data import Dataset

class CycloneTracksDataset(Dataset):

    def __init__(self, ds_name, data_dir, train, window_size=4, dims=(1, 2, 3), min_=None, max_=None):
        # Initialize
        super(CycloneTracksDataset, self).__init__()
        self.ds_name = ds_name
        self.window_size = window_size
        self.dims = dims
        if train:
            self.data_split = 'train'
        else:
            self.data_split = 'test'

        # Load data config
        config_yml_path = os.path.join(data_dir, 'config.yml')
        with open(config_yml_path, 'r') as config_file:
            self.data_config = yaml.safe_load(config_file)[self.ds_name]

        # Load mat file
        data_path = os.path.join(data_dir, self.data_config[self.data_split])
        self.data = sio.loadmat(data_path)[f'cyclones_{self.data_split}'][0]

        # Min-max
        self.min, self.max = min_, max_

        # Extract tensors from data
        self.X, self.y, self.track_id = self.extract_data(self.data)
    
    def create_sequences(self, X, track_id):
        # Create empyty sequences
        Xs, ys = [], []

        # Filter selected dims
        X = X[:, self.dims]

        # Scale the data
        X = self.normalize(X)

        # Add sequences to Xs and ys
        for i in range(len(X)-self.window_size):
            Xs.append(X[i: (i + self.window_size)])
            ys.append(X[i + self.window_size])

        # Track id
        track_ids = np.full(len(Xs), track_id)

        return np.array(Xs), np.array(ys), track_ids
        
    def extract_data(self, data):

        # Min-max scaling
        data_stacked = np.concatenate(data, axis=0)[:, self.dims]
        if self.min is None:
            self.min = data_stacked.min(axis=0)
            self.max = data_stacked.max(axis=0)

        # Scale and create sequences
        data = [self.create_sequences(data[track_idx], track_id=track_idx+1) for track_idx in range(len(data))]
        X, y, track_id = list(zip(*data))

        # Concatenate tracks
        X = torch.tensor(np.concatenate(X, axis=0)).type(torch.float)
        y = torch.tensor(np.concatenate(y, axis=0)).type(torch.float)
        track_id = torch.tensor(np.concatenate(track_id, axis=0)).type(torch.int)
        
        return X, y, track_id

    def denormalize(self, X):
        """Unscale data to orginal scale from 0-1
        """
        return X * (self.max - self.min) + self.min

    def normalize(self, X):
        """Scale data between 0 and 1
        """
        return (X - self.min) / (self.max - self.min)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index], self.track_id[index]



if __name__ == "__main__":
    root_dir = '../variational-rnn-cyclones/cyclone_data/'
    ds_name = 'north_indian_ocean'

    ds = CycloneTracksDataset(ds_name, root_dir, train=False)
    ds[0]
