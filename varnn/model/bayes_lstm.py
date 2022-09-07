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
        
    def forward(self, x, h, c, sampling=False):

        # Combine input and hidden
        combined = torch.cat((x, h), 1)
        
        # Weights for memory and hidden state update
        i_t = torch.sigmoid(self.input_gate(combined))
        f_t = torch.sigmoid(self.forget_gate(combined))
        c_hat_t = torch.tanh(self.cell_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        
        # Update memory
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_hat_t) 
        
        # Update hidden state
        h_t = torch.mul(o_t, torch.tanh(c_t))
        
        # Output layer
        out = self.output(h_t)

        return out, h_t, c_t

    
    def init_zero_hidden(self, batch_size=1) -> torch.Tensor:
        """
                Helper function.
        Returns a hidden state with specified batch size. Defaults to 1
        """
        h_t = nn.init.kaiming_normal_(
                    torch.empty((batch_size, self.self.hidden_dim))
            )
        c_t = nn.init.kaiming_normal_(
                    torch.empty((batch_size, self.self.hidden_dim))
            )

        return h_t, c_t

    
    def log_prior(self):
        return (
            self.input_gate.log_prior + 
            self.forget_gate.log_prior + 
            self.cell_gate.log_prior +
            self.output_gate.log_prior +
            self.output
        )
    
    def log_variational_posterior(self):
        return (
            self.input_gate.log_variational_posterior + 
            self.forget_gate.log_variational_posterior + 
            self.cell_gate.log_variational_posterior +
            self.output_gate.log_variational_posterior +
            self.output
        )
