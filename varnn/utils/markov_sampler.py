from __future__ import absolute_import

import torch
import torch.nn as nn

from ..config import constants


class MarkovSamplingLoss(object):

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
        log_prior = torch.tensor(0, dtype=torch.float)
        log_variational_posterior = torch.tensor(0, dtype=torch.float)

        # Sample and compute pdfs
        for s in range(self.samples):

            # Initialize hidden state
            h_t = self.model.init_zero_hidden(batch_size=X.shape[0])

            for t in range(seq_size):
                o_t, h_t = self.model(X[:, t], h_t.detach(), sampling=True)

            outputs[s] = o_t
            
            if testing:
                continue
            
            log_prior += self.model.log_prior()
            log_variational_posterior += self.model.log_variational_posterior()
        
        # Return output if testing
        if testing:
            return outputs

        # Log prior, variational posterior and likelihood
        negative_log_likelihood = self.mse(outputs.mean(0), y)
        loss = (log_variational_posterior - log_prior)/num_batches + negative_log_likelihood
        
        return loss, outputs
