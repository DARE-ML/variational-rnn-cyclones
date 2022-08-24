from __future__ import absolute_import

import torch
import torch.nn as nn

from .weight import GaussianWeight, ScaledMixedGaussian
from ..config import constants

class BayesLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        
        # Weights
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(
                            torch.Tensor(
                                self.in_features, 
                                self.out_features
                            ).uniform_(-0.2, 0.2)
                        )
        self.bias_mu = nn.Parameter(
                            torch.Tensor(
                                1, 
                                self.out_features
                            ).uniform_(-0.2, 0.2)
                        )
        
        self.weight_rho = nn.Parameter(
                            torch.Tensor(
                                self.in_features, 
                                self.out_features
                            ).uniform_(-5, -4)
                        )
        
        self.bias_rho  = nn.Parameter(
                            torch.Tensor(
                                1, 
                                self.out_features
                            ).uniform_(-5, -4)
                        )
    
        self.weight = GaussianWeight(self.weight_mu, self.weight_rho)
        self.bias = GaussianWeight(self.bias_mu, self.bias_rho)
         
        # Prior
        self.prior_weight = ScaledMixedGaussian(
                                constants.PI, 
                                constants.SIGMA1, 
                                constants.SIGMA2
                            )

        self.prior_bias = ScaledMixedGaussian(
                                constants.PI, 
                                constants.SIGMA1,
                                constants.SIGMA2
                            )
        
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, x, sampling=False, calculate_log_probs=False):
        
        if self.training or sampling:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        
        if self.training or calculate_log_probs:
            self.log_prior = self.prior_weight.log_prob(weight) + self.prior_bias.log_prob(bias)    
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        
        Y = torch.matmul(x, weight) + bias
       
        return Y