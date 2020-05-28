# %%

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import random
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import json
from sklearn.manifold import TSNE

#%% Define paths

data_root_dir = '~/Downloads/datasets'


#%% Create dataset

"""train_transform = transforms.Compose([
    transforms.ToTensor(),
])"""

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

#train_dataset = MNIST(data_root_dir, train=True,  download=True, transform=train_transform)
test_dataset  = MNIST(data_root_dir, train=False, download=True, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

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

# %% create autoencoder with best params

model_params = json.load(open("best_params.json"))
net = Autoencoder(encoded_space_dim=model_params["hidden"])
net.load_state_dict(torch.load('net_params.pth', map_location=torch.device('cpu')))

# %% predi
def predict(net, dataloader, corruption=None, corruption_level=0.1):

    net.eval()
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad(): # No need to track the gradients
        original  = torch.Tensor().float()
        img  = torch.Tensor().float()
        pred = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0]
            if corruption == "noise":
                # save original image
                original = torch.cat([original, image_batch])
                # add noise
                corruption = corruption_level * torch.randn(*image_batch.shape)
                image_batch = image_batch + corruption
                image_batch = torch.clamp(image_batch, 0., 1.)
            elif corruption == "occlusion":
                # save original image
                original = torch.cat([original, image_batch])
                # add occlusion
                corruption = np.random.choice([0.,1.], size=image_batch.shape, p=[corruption_level, 1.-corruption_level])
                image_batch = image_batch * torch.FloatTensor(corruption)
                image_batch = torch.clamp(image_batch, 0., 1.)
            elif corruption is None:
                original = torch.cat([original, image_batch])
                pass
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            img  = torch.cat([img, image_batch])
            pred = torch.cat([pred, out])

        # Evaluate global loss
        loss = float(loss_fn(pred, original))

    if corruption is None:
        return loss, img.numpy(), pred.numpy()
    else:
        return loss, img.numpy(), pred.numpy(), original.numpy()

# %% predictions on original images
loss, img, out = predict(net, test_dataloader, corruption=None)
print("Loss with original images = {}".format(loss))

# %% plotting function
def plot_corruption(original, feed, out, fname=None):
    fig, axs = plt.subplots(1, 3, figsize=(9,3))
    axs[0].imshow(original.squeeze(), cmap='gist_gray')
    axs[0].set_title("Original Image", fontsize=14)
    axs[0].axis("off")
    axs[1].imshow(feed.squeeze(), cmap='gist_gray')
    axs[1].set_title("Autoencoder Input", fontsize=14)
    axs[1].axis("off")
    axs[2].imshow(out.squeeze(), cmap='gist_gray')
    axs[2].set_title("Autoencoder Output", fontsize=14)
    axs[2].axis("off")
    plt.tight_layout()
    if fname is not None: fig.savefig(fname)
    plt.close()
    return

# %% noisy images

corruption = "noise"
corruption_level = np.arange(0., 1., 0.1)
loss_list = []
os.makedirs(corruption, exist_ok=True)
for i, c in enumerate(corruption_level):
    loss, img, out, original = predict(net, test_dataloader, corruption=corruption, corruption_level=c)
    loss_list.append(loss)
    idx = 13
    fname = "{}/{:02d}.pdf".format(corruption, i)
    plot_corruption(original[idx], img[idx], out[idx], fname)

plt.figure(figsize=(8,5))
plt.grid()
plt.plot(corruption_level, loss_list, "o--")
plt.xlabel("Noise level", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig("{}/loss.pdf".format(corruption))

# %% occluded images

corruption = "occlusion"
corruption_level = np.arange(0., 1., 0.1)
loss_list = []
os.makedirs(corruption, exist_ok=True)
for i, c in enumerate(corruption_level):
    loss, img, out, original = predict(net, test_dataloader, corruption=corruption, corruption_level=c)
    loss_list.append(loss)
    idx = 100
    fname = "{}/{:02d}.pdf".format(corruption, i)
    plot_corruption(original[idx], img[idx], out[idx], fname)

plt.figure(figsize=(8,5))
plt.grid()
plt.plot(corruption_level, loss_list, "o--")
plt.xlabel("Noise level", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig("{}/loss.pdf".format(corruption))

# %% Get the encoded representation of the test samples
encoded_image = []
encoded_label = []
for sample in tqdm(test_dataset):
    img = sample[0].unsqueeze(0)
    label = sample[1]
    # Encode image
    net.eval()
    with torch.no_grad():
        encoded_img = net.encode(img)
    # Append to list
    encoded_image.append(encoded_img.flatten().numpy())
    encoded_label.append(label)

encoded_image = np.array(encoded_image)
tsne   = TSNE(n_components=2, init='pca', random_state=0, n_jobs=-1)
X_tsne = tsne.fit_transform(encoded_image)

encoded_samples = [(X_tsne[x], encoded_label[x]) for x in range(len(encoded_label))]

# %% Visualize encoded space
color_map = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
        6: '#e377c2',
        7: '#7f7f7f',
        8: '#bcbd22',
        9: '#17becf'
        }

# Plot just 1k points
encoded_samples_reduced = random.sample(encoded_samples, 2000)
plt.figure(figsize=(10,10))
for enc_sample, label in tqdm(encoded_samples_reduced):
    plt.plot(enc_sample[0], enc_sample[1], marker='.', color=color_map[label])
plt.grid(True)
plt.tick_params(labelbottom=False, labelleft=False)
plt.legend([plt.Line2D([0], [0], ls='', marker='.', color=c, label=l) for l, c in color_map.items()], color_map.keys())
plt.tight_layout()
plt.savefig("encoded_space.pdf")
plt.show()


# %%
