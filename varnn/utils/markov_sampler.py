from __future__ import absolute_import

import torch
import torch.nn as nn

from ..config import constants


class MarkovSampler(object):

    def __init__(self, model, samples=constants.SAMPLES) -> None:
        self.model = model
        self.samples = samples
        self.mse = nn.MSELoss()

    def __call__(self, X, y, num_batches, testing=False):
        
        # Validate input shape
        assert len(X.shape)==3, f"Expected input to be 3-dim, got {len(X.shape)}"
        batch_size, seq_size, feat_size = X.shape

        # Define output tensors
        outputs = torch.zeros(self.samples, batch_size, self.model.output_dim)
        log_priors = torch.zeros(self.samples)
        log_variational_posterior = torch.zeros(self.samples)
        
        # Sample and compute pdfs
        for s in range(self.samples):
            outputs[s] = self.model(X, sampling=True)
            if testing:
                continue
            log_priors[s] = self.model.log_prior()
            log_variational_posterior[s] = self.model.log_variational_posterior()
        
        # Return if testing
        if testing:
            return outputs

        # Log prior, variational posterior and likelihood
        log_prior = log_priors.sum()
        log_variational_posterior = log_variational_posterior.sum()
        negative_log_likelihood = mse_loss = self.mse(outputs.mean(0), y)
        loss = (log_variational_posterior - log_prior + negative_log_likelihood)/num_batches
        
        return loss, mse_loss, outputs
