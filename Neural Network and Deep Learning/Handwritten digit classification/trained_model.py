# %% load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("default")

import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F

#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#print("DEVICE:", device)

# %% load data
mat = scipy.io.loadmat('MNIST.mat')
# input images
X = mat['input_images'] # np.array, shape=(60000 x 784)
                        # each row is a single image
                        # each column is a pixel of the image (28 x 28)
# labels
Y = np.squeeze(mat['output_labels']).astype(int)

X = torch.from_numpy(X)
Y = torch.from_numpy(Y)

#%% Neural Network

### Define the network class
class Net(nn.Module):

    def __init__(self,
                 Ni  = 784,
                 Nh1 = 256,
                 Nh2 = 256,
                 No  = 10,
                 act = nn.ReLU):
        super().__init__()

        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)

        self.act = act()

    def forward(self, x, additional_out=False):

        x   = self.act(self.fc1(x))
        x   = self.act(self.fc2(x))
        out = self.fc3(x)

        if additional_out:
            return out, x

        return out

# %% load trained model
net = Net(Nh1=280, Nh2=121)
net.load_state_dict(torch.load("best_model.torch"))

# %% make predictions
net.eval()
with torch.no_grad():
    y_pred = net(X)
y_pred = F.log_softmax(y_pred, dim=1)
_, y_class = y_pred.max(1)
np_Y = Y.cpu().numpy()
np_y_pred = y_class.cpu().numpy()
correct = np.sum(np_Y == np_y_pred)
print("Accuracy = {:.5f} %".format(correct*100/np_Y.shape[0]))
