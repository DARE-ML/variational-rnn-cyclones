import torch

from varnn.model.bayes_rnn import BayesRNN
from varnn.utils.markov_sampler import MarkovSampler

# Dimensions
input_dim = 3
hidden_dim = 16
output_dim = 3

# Model
brnn = BayesRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Input
batch_size = 10
seq_len = 5
X = torch.rand(batch_size, seq_len, input_dim)
y = torch.rand(batch_size, output_dim)

# Predict
sampler = MarkovSampler(brnn)

# During Training
loss, mse_loss, outputs = sampler(X, y, num_batches=1, testing=False)
print(loss)
print(mse_loss)
print(outputs.shape)

# During Testing
outputs = sampler(X, y, num_batches=1, testing=True)
print(outputs.shape)
