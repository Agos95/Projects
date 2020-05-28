# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn
from gensim.models import Word2Vec


class Network(nn.Module):

    def __init__(self, input_size, hidden_units, layers_num, embedding, dropout_prob=0):
        # Call the parent init function (required!)
        super().__init__()
        # Embedding
        model = Word2Vec.load('w2v.model')
        weights = torch.FloatTensor(model.wv.vectors)
        self.embeddings = nn.Embedding.from_pretrained(weights)
        self.embeddings.weight.requires_grad = False
        # Define recurrent layer
        self.rnn = nn.LSTM(input_size=embedding,
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
        # Define output layer
        self.out = nn.Linear(hidden_units, input_size)

    def forward(self, x, state=None):
        # Embedding
        x = self.embeddings(x)
        # LSTM
        x, rnn_state = self.rnn(x, state)
        # Linear layer
        x = self.out(x)
        return x, rnn_state


def train_batch(net, dataloader, loss_fn, optimizer, device):

    # training
    net.train()
    batch_train_loss = []

    for batch_sample in dataloader:
        # Extract batch
        batch = batch_sample['encoded'].to(device)

        ### Prepare network input and labels
        # Get the labels (from second to last last letter of each sequence)
        labels = batch[:, 1:]
        # Remove the labels from the input tensor
        net_input = batch[:, :-1]

        ### Forward pass
        # Eventually clear previous recorded gradients
        optimizer.zero_grad()
        # Forward pass
        net_out, _ = net(net_input)

        ### Update network
        # Evaluate loss only for last output
        loss = loss_fn(net_out.transpose(1, 2), labels)
        # Backward pass
        loss.backward()
        # Update
        optimizer.step()
        batch_train_loss.append(float(loss.data))

    # Return average batch loss
    return np.mean(batch_train_loss)

def test_batch(net, dataloader, loss_fn, optimizer, device):

    # test
    net.eval()
    batch_test_loss = []
    with torch.no_grad():
        for batch_sample in dataloader:
            # Extract batch
            batch = batch_sample['encoded'].to(device)
            ### Prepare network input and labels
            # Get the labels (the last letter of each sequence)
            labels = batch[:, 1:]
            # Remove the labels from the input tensor
            net_input = batch[:, :-1]

            ### Forward pass
            net_out, _ = net(net_input)

            ### Update network
            # Evaluate loss only for last output
            loss = loss_fn(net_out.transpose(1, 2), labels)
            batch_test_loss.append(float(loss.data))

    # Return average batch loss
    return np.mean(batch_test_loss)


if __name__ == '__main__':

    #%% Get some real input from dataset

    from torch.utils.data import DataLoader
    from dataset import LoadDataset, ToTensor
    from torchvision import transforms

    filepath = 'Scienza_in_cucina.txt'
    crop_len = 20
    embedding = 100
    trans = ToTensor()
    dataset = LoadDataset(filepath, transform=trans, crop_len=crop_len, embedding=embedding)

    dataloader = DataLoader(dataset, batch_size=52, shuffle=True)

    for batch_sample in dataloader:
        batch_onehot = batch_sample['encoded']
        print(batch_onehot.shape)

    #%% Initialize network
    input_size = dataset.vocabulary_length()
    hidden_units = 128
    layers_num = 2
    dropout_prob = 0.3
    net = Network(input_size, hidden_units, layers_num, embedding, dropout_prob)


    #%% Test the network output

    out, rnn_state = net(batch_onehot)
    print(out.shape)
    print(rnn_state[0].shape)
    print(rnn_state[1].shape)

    #%% Test network update

    import torch
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()))
    loss_fn = nn.CrossEntropyLoss()

    train_batch(net, batch_onehot, loss_fn, optimizer)

