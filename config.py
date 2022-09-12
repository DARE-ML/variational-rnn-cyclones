import os
from argparse import ArgumentParser

from varnn.config import constants

# Define the parser
parser = ArgumentParser(description="Train model")

# Add the parser parameters
parser.add_argument("--ds-name", type=str, required=True)
parser.add_argument("--model", 
                    type=str, default="brnn", 
                    help="Model to train: rnn, brnn, lstm, blstm"
                    )
parser.add_argument("--lr", 
                    type=float, default=0.01, 
                    help="Learning rate to be used for model training"
                    )
parser.add_argument("--epochs", 
                    type=int, default=100, 
                    help="Number of epochs to train the model for"
                    )
parser.add_argument("--batch-size", 
                    type=int, default=1024, 
                    help="Number of sequences in a single training batch"
                    )
parser.add_argument("--samples", 
                    type=int, default=constants.SAMPLES, 
                    help="Number of markov samples used for loss computation"
                    )
parser.add_argument("--hidden", 
                    type=int, default=32, 
                    help="Number of hidden features in the model architecture"
                    )
parser.add_argument("--root-dir", 
                    type=str, default=os.getcwd(), 
                    help="Path to the directory where the repository is located"
                    )
parser.add_argument("--features", 
                    type=str, default="location", 
                    help="Features to train model on: can be location, intensity or both"
                    )


# Parse Arguments
opt = parser.parse_args()


# Assign dimensions based on features
if opt.features == "location":
    opt.dims = (1, 2)
elif opt.features == "intensity":
    opt.dims = (3)
elif opt.features == "both":
    opt.dims = (1, 2, 3)
else:
    raise(ValueError("Value of features must be either `location`, `intensity`, or `both`"))



