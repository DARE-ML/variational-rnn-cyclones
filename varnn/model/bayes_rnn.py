from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np

from ..utils.layers import BayesLinear
from ..config import constants


class BayesRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super().__init__()

        # Input Dims
        self.input_dim = input_dim

        # Hidden Dims
        self.hidden_dim = hidden_dim

        # Output Dims
        self.output_dim = output_dim
        
        # Weights
        self.w_ih = BayesLinear(input_dim, hidden_dim)
        self.w_hh = BayesLinear(hidden_dim, hidden_dim)
        self.w_ho = BayesLinear(hidden_dim, output_dim)
        
    def forward(self, x, sampling=False):
        
        # Validate input shape
        assert len(x.shape)==3, f"Expected input to be 3-dim, got {len(x.shape)}"

        # Get dimensions of the input
        batch_size, seq_size, input_size = x.shape

        # Initialize hidden_state
        h_t = self.init_zero_hidden(batch_size)

        for t in range(seq_size):
            
            # Input at time step t
            x_t = x[:, t, :]

            # Hidden state
            h_t = torch.tanh(self.w_ih(x_t, sampling) + self.w_hh(h_t, sampling))

        # Update output
        o = torch.tanh(self.w_ho(h_t, sampling))

        return o
    
    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
                Helper function.
        Returns a hidden state with specified batch size. Defaults to 1
        """
        return torch.zeros(batch_size, self.hidden_dim, requires_grad=False)

    
    def log_prior(self):
        return (
            self.w_ih.log_prior + 
            self.w_hh.log_prior + 
            self.w_ho.log_prior
        )
    
    def log_variational_posterior(self):
        return (
            self.w_ih.log_variational_posterior + 
            self.w_hh.log_variational_posterior +
            self.w_ho.log_variational_posterior
        )