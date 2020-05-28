# %%
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
device  = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("device: ", device)
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from sklearn.model_selection import KFold
import json


#%% Define paths

data_root_dir = '~/Downloads/datasets'


#%% Create dataset

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = MNIST(data_root_dir, train=True,  download=True, transform=train_transform)
test_dataset  = MNIST(data_root_dir, train=False, download=True, transform=test_transform)


#%% Define the network architecture

class Autoencoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )

        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x

    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

#%% Network training

### Training function
def train_epoch(net, dataloader, loss_fn, optimizer):
    # Training
    net.train()
    loss_log = []
    for sample_batch in dataloader:
        # Extract data and move tensors to the selected device
        image_batch = sample_batch[0].to(device)
        # Forward pass
        output = net(image_batch)
        loss = loss_fn(output, image_batch)
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_log.append(float(loss))
        # Print loss
        #print('\t partial train loss: %f' % (loss.data))
    return np.mean(loss_log)


### Testing function
def test_epoch(net, dataloader, loss_fn, optimizer):
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()])
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return float(val_loss)#val_loss.data

### Kfold cross validation
def kfoldCV(net, optimizer, dataset, loss_fn, nfolds=5, epochs=1000, rs=100, batch_size=512, verbose=False):
    kf = KFold(nfolds, shuffle=True, random_state=rs)
    current_Fold = 1
    train_losses = []
    validation_losses = []
    for train_index, test_index in kf.split(dataset):
        if verbose:
            print("{}CV split {} / {}".format(" "*5, current_Fold, nfolds))

        # get train/validation folds
        train_dataset = Subset(dataset, train_index)
        test_dataset  = Subset(dataset, test_index)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        t_loss, v_loss = [], []
        for _ in range(epochs):
            ### Training
            tr_loss = train_epoch(net, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim)
            ### Validation
            val_loss = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim)

            t_loss.append(tr_loss)
            v_loss.append(val_loss)

        train_losses.append(np.mean(t_loss[-5:]))
        validation_losses.append(np.mean(v_loss[-5:]))
        current_Fold += 1


    validation_loss = np.mean(validation_losses)
    train_loss      = np.mean(train_losses)

    return train_loss, validation_loss

# %% tune learning rate and weight decay of the optimizer

print("="*50)
print("{:^50}".format("Tuning the optimizer"))
print("="*50)

### Define a loss function
loss_fn = torch.nn.MSELoss()

### Define optimizer parameters search
lr_list = [1e-02, 1e-03, 1e-04] + [5e-02, 5e-03, 5e-04] # Learning rate
weight_decay_list = [1e-04, 1e-05, 1e-06] + [5e-04, 5e-05, 5e-06]
encoded_space_dim = 6


# iterations indicators
total_cycles = len(lr_list)*len(weight_decay_list)
it = 0

# dataframe to store results
results = pd.DataFrame(columns=["lr", "wd", "Train_Loss", "Val_Loss"])

pbar = tqdm(total=len(lr_list)*len(weight_decay_list))

for lr in lr_list:
    for weight_decay in weight_decay_list:
        #it += 1
        #print("Iteration {:2d} / {:2d}".format(it, total_cycles))
        net = Autoencoder(encoded_space_dim=encoded_space_dim)
        net.to(device)
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss, validation_loss = kfoldCV(net, optim, train_dataset, loss_fn, nfolds=3, epochs=20, verbose=False)
        log = pd.DataFrame({
            "lr" : [lr],
            "wd" : [weight_decay],
            "Train_Loss" : [train_loss],
            "Val_Loss"   : [validation_loss]
        })
        results = results.append(log)

        pbar.update()
pbar.close()

results.sort_values(by="Val_Loss", inplace=True)

print("Parameters search over optimizer results:\n", results, "\n\n")
results.to_csv("Optmizer_csv.csv"    , index=False)
results.to_latex("Optmizer_latex.txt", index=False)

best = results.iloc[0].to_dict()
lr   = best["lr"]
wd   = best["wd"]

# %% train the model changing the size of hidden layer

print("="*50)
print("{:^50}".format("Tuning the hidden layer dimension"))
print("="*50)

### Define dataloader
train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=512, shuffle=False)
# hidden layer size
encoded_space_dim = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]

# parameters
epochs = 50
loss_fn = torch.nn.MSELoss()

results = pd.DataFrame(columns=["esd", "Test_Loss"])
train_loss_logs = {str(esd):[] for esd in encoded_space_dim}
test_loss_logs  = {str(esd):[] for esd in encoded_space_dim}

min_loss = 1e06

for esd in tqdm(encoded_space_dim):
    net = Autoencoder(esd)
    net.to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optim, factor=0.75, patience=5)
    tr_loss, t_loss = [], []
    for ep in range(epochs):
        ### Training
        train_loss = train_epoch(net, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim)
        tr_loss.append(train_loss)
        ### Validation
        test_loss = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim)
        t_loss.append(test_loss)
        # update optimizer scheduler
        scheduler.step(test_loss)
    current_loss = np.mean(t_loss[-5:])
    log = pd.DataFrame({
        "esd" : [esd],
        "Test_Loss" : [current_loss]
    })
    results = results.append(log)
    train_loss_logs[str(esd)] = tr_loss
    test_loss_logs[str(esd)]  = t_loss

    if current_loss < min_loss:
        # Save network parameters
        torch.save(net.state_dict(), 'net_params.pth')
        min_loss = current_loss

results.sort_values(by="Test_Loss", inplace=True)

print("\nParameters search over hidden layer size results:\n", results, "\n\n")
results.to_csv("Hidden_csv.csv"    , index=False)
results.to_latex("Hidden_latex.txt", index=False)
best = results.iloc[0].to_dict()

# %% Save best parameters

final_param = {
    "hidden" : best["esd"],
    "lr"     : lr,
    "wd"     : wd
}
with open('best_params.json', 'w') as f:
    json.dump(final_param, f, indent=4)

np.savez_compressed("train_hidden_loss.npz", **train_loss_logs)
np.savez_compressed("test_hidden_loss.npz" , **test_loss_logs)

# %% plot test loss for different hidden sizes
plt.figure(figsize=(8,5))
plt.grid()
for esd, loss in test_loss_logs.items():
    plt.plot(loss, label=esd)
plt.legend(title="Hidden layer size", title_fontsize=12)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Test Loss", fontsize=14)
plt.savefig("hidden_loss.pdf")
plt.close()

# %% plot train/test loss for best model
plt.figure(figsize=(8,5))
plt.grid()
plt.plot(train_loss_logs[str(best["esd"])], label="Train Loss")
plt.plot( test_loss_logs[str(best["esd"])], label="Test Loss")
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=14)
plt.savefig("train-test_loss_best.pdf")
plt.close()
