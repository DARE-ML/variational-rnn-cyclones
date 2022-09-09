# VaRNN - Cyclone Track Prediction
VaRNN provides implementation of Variational Bayes algorithm (inspired by bayes by backprop) for RNN models including Long Short-term Memory models. We apply these models for prediction of cyclone tracks in data provided by Joint Typhoon Warning Center (JTWC).

## Required Packages
Requires Python 3.7 or later with PyTorch and related libraries. Please refer to requirements.txt for details of python packages required

## Training the model
The `train.py` file provides arguments for training the BayesRNN and BayesLSTM models respectively. Here are the details of training arguments available:
```{bash}
(py39) ➜  variational-rnn-cyclones (main) python train.py --help                                           ✭ ✱
usage: train.py [-h] --ds-name DS_NAME [--model MODEL] [--lr LR] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--samples SAMPLES]
                [--hidden HIDDEN] [--root-dir ROOT_DIR] [--features FEATURES]

Train model

optional arguments:
  -h, --help            show this help message and exit
  --ds-name DS_NAME
  --model MODEL         Model to train: rnn, varnn, lstm, varlstm
  --lr LR               Learning rate to be used for model training
  --epochs EPOCHS       Number of epochs to train the model for
  --batch-size BATCH_SIZE
                        Number of sequences in a single training batch
  --samples SAMPLES     Number of markov samples used for loss computation
  --hidden HIDDEN       Number of hidden features in the model architecture
  --root-dir ROOT_DIR   Path to the directory where the repository is located
  --features FEATURES   Features to train model on: can be location, intensity or both
```




