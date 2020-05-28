# -*- coding: utf-8 -*-

from numpy.random import seed as RNGseed
import json
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import argparse
from network import Network
from dataset import encode_text, decode_text
from pathlib import Path


##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Generate a chapter starting from a given text')

parser.add_argument('--seed',      type=str, default='pasta al sugo', help='Initial text of the chapter')
parser.add_argument('--model_dir', type=str, default='BestModel',     help='Network model directory')
parser.add_argument('--length',    type=int, default=50,              help='Number of predicted words to print')
parser.add_argument('--random',    type=int, default=None,            help='RNG seed for predictions')

##############################
##############################
##############################

if __name__ == '__main__':

    ### Parse input arguments
    args = parser.parse_args()
    # set RNG seed
    RNGseed(args.random)

    #%% Load training parameters
    model_dir = Path(args.model_dir)
    print ('Loading model from: %s' % model_dir)
    training_args = json.load(open(model_dir / 'training_args.json'))

    #%% Load encoder and decoder dictionaries
    number_to_char = json.load(open(model_dir / 'number_to_char.json'))
    char_to_number = json.load(open(model_dir / 'char_to_number.json'))

    #%% Initialize network
    net = Network(input_size  =training_args["input_size"],
                  hidden_units=training_args["hidden_units"],
                  layers_num  =training_args["layers_num"],
                  embedding   =training_args["embedding_dim"],
                  dropout_prob=training_args["dropout_prob"])

    #%% Load network trained parameters
    net.load_state_dict(torch.load(model_dir / 'net_params.pth', map_location='cpu'))
    net.eval() # Evaluation mode (e.g. disable dropout)

    #%% Find initial state of the RNN
    with torch.no_grad():
        # print the seed
        print("="*50)
        print("SEED: {}".format(args.seed))
        print("="*50, "\n")
        # Encode seed
        seed_encoded = encode_text(char_to_number, args.seed.split())
        # To tensor
        seed_tensor = torch.LongTensor(seed_encoded)
        # Add batch axis
        seed_tensor = seed_tensor.unsqueeze(0)
        # Forward pass
        net_out, net_state = net(seed_tensor)
        # Sample the last output accordingly to sofmax probability
        net_out = F.softmax(net_out[:, -1, :], dim=1)
        next_char_encoded = int(Categorical(net_out).sample())
        # Print the seed letters
        print(args.seed, end='', flush=True)
        next_char = number_to_char[str(next_char_encoded)]
        print(next_char, end='', flush=True)

    #%% Generate chapter
    for n in range(1, args.length):
        with torch.no_grad(): # No need to track the gradients
            # The new network input is the last chosen letter
            net_input = torch.LongTensor([next_char_encoded])
            net_input = net_input.unsqueeze(0)
            # Forward pass
            net_out, net_state = net(net_input, net_state)
            # Sample the next output accordingly to sofmax probability
            net_out = F.softmax(net_out[:, -1, :], dim=1)
            next_char_encoded = int(Categorical(net_out).sample())
            # Decode the letter
            next_char = number_to_char[str(next_char_encoded)]
            print(next_char, end='', flush=True)
    print("\n")












