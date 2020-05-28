"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering
PHYSICS OF DATA - Department of Physics and Astronomy
COGNITIVE NEUROSCIENCE AND CLINICAL NEUROPSYCHOLOGY - Department of Psychology

A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti

Author: Federico Agostini

Lab. 03 - Handwritten digits classification

"""

# https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
# https://nextjournal.com/gkoehler/pytorch-mnist
# %% import modules
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("default")
import scipy.io
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# %% load data
mat = scipy.io.loadmat('./MNIST.mat')
# input images
X = mat['input_images'] # np.array, shape=(60000 x 784)
                        # each row is a single image
                        # each column is a pixel of the image (-> need to be reshaped to 28 x 28)
X = X.reshape((X.shape[0], 1, 28, 28))
# labels
Y = np.squeeze(mat['output_labels']).astype(int)

# %% train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1994)

X_train = torch.from_numpy(X_train)
Y_train = torch.from_numpy(Y_train)
X_test  = torch.from_numpy(X_test)
Y_test  = torch.from_numpy(Y_test)

print("shape X_train: ",np.shape(X_train))
print("shape Y_train: ",np.shape(Y_train))
print("shape X_test: " ,np.shape(X_test))
print("shape Y_test: " ,np.shape(Y_test))

# %% neural network class

class Net(nn.Module):

    def __init__(self):
        # to "enable" inheritace
        super().__init__()
        """
        class Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        """
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128) # 9216 = 64*12*12
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        #x = torch.flatten(x, 1)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# %% create network and define parameters

n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
#torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
loss = nn.CrossEntropyLoss()

"""if torch.cuda.is_available():
    network = network.cuda()
    loss = loss.cuda()
"""
# %% train the model

def train(epoch):
    network.train()
    tr_loss = 0
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()

    # getting the training set
    x_train, y_train = X_train, Y_train
    # getting the validation set
    x_test, y_test   = X_test, Y_test

    """if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_test  = x_test.cuda()
        y_test  = y_test.cuda()
"""

    # prediction for training and test set
    output_train = network(x_train)
    output_test  = network(x_test)

    # computing the training and validation loss
    loss_train = loss(output_train, y_train)
    loss_test  = loss(output_test, y_test)
    train_losses.append(loss_train)
    test_losses.append(loss_test)

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_test)

# %%
# defining the number of epochs
n_epochs = 25
# empty list to store training losses
train_losses = []
# empty list to store validation losses
test_losses = []
# training the model
for epoch in range(n_epochs):
    train(epoch)



# %%
