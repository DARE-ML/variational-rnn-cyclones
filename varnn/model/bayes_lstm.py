from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np

from ..utils.layers import BayesLinear
from ..config import constants


class BayesLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        
        super().__init__()
        
        # Intialize dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Define Gates
        self.input_gate = BayesLinear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.forget_gate = BayesLinear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.cell_gate = BayesLinear(self.input_dim + self.hidden_dim, self.hidden_dim)
        self.output_gate = BayesLinear(self.input_dim + self.hidden_dim,self.hidden_dim)
        self.output = BayesLinear(self.hidden_dim, self.output_dim)
        
    def forward(self, x, sampling=False):

        # Validate input shape
        assert len(x.shape)==3, f"Expected input to be 3-dim, got {len(x.shape)}"
        # Get dimensions of the input
        batch_size, seq_size, input_size = x.shape

        # Initialize hidden state and memory
        h_t, c_t = self.init_zero_hidden(batch_size)

        for t in range(seq_size):

            # Input at time step t
            x_t = x[:, t, :]

            # Combine input and hidden
            combined = torch.cat((x_t, h_t), 1)
            
            # Weights for memory and hidden state update
            i_t = torch.sigmoid(self.input_gate(combined))
            f_t = torch.sigmoid(self.forget_gate(combined))
            c_hat_t = torch.tanh(self.cell_gate(combined))
            o_t = torch.sigmoid(self.output_gate(combined))
            
            # Update memory
            c_t = torch.mul(f_t, c_t) + torch.mul(i_t, c_hat_t) 
            
            # Update hidden state
            h_t = torch.mul(o_t, torch.tanh(c_t))
            
        # Output layer
        o = self.output(h_t)

        return o

    
    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
                Helper function.
        Returns a hidden state with specified batch size. Defaults to 1
        """
        h_t = nn.init.kaiming_normal_(
                    torch.empty((batch_size, self.hidden_dim))
            )
        c_t = nn.init.kaiming_normal_(
                    torch.empty((batch_size, self.hidden_dim))
            )

        return h_t, c_t

    
    def log_prior(self):
        return (
            self.input_gate.log_prior + 
            self.forget_gate.log_prior + 
            self.cell_gate.log_prior +
            self.output_gate.log_prior +
            self.output.log_prior
        )
    
    def log_variational_posterior(self):
        return (
            self.input_gate.log_variational_posterior + 
            self.forget_gate.log_variational_posterior + 
            self.cell_gate.log_variational_posterior +
            self.output_gate.log_variational_posterior +
            self.output.log_variational_posterior
        )
