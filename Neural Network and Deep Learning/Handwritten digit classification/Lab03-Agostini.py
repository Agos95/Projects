#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("default")
from tqdm import trange, tqdm
from IPython.display import display

import scipy.io
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", device)

# %% load data
mat = scipy.io.loadmat('./MNIST.mat')
# input images
X = mat['input_images'] # np.array, shape=(60000 x 784)
                        # each row is a single image
                        # each column is a pixel of the image (28 x 28)
# labels
Y = np.squeeze(mat['output_labels']).astype(int)

# %% Print some examples
# take 4 random images
np.random.seed(759045)
idx = np.random.randint(0, X.shape[0]+1, 4)
plt.figure()
for i,img in enumerate(idx):
    plt.subplot(2,2,i+1)
    plt.axis("off")
    plt.imshow(X[img].reshape(28,28).T, cmap="gray")
    plt.title(Y[img])

# %% train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1995)

X_train = torch.from_numpy(X_train).to(device)
Y_train = torch.from_numpy(Y_train).to(device)
X_test  = torch.from_numpy(X_test).to(device)
Y_test  = torch.from_numpy(Y_test).to(device)

print("shape X_train: ",np.shape(X_train))
print("shape Y_train: ",np.shape(Y_train))
print("shape X_test: " ,np.shape(X_test))
print("shape Y_test: " ,np.shape(Y_test))

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

# %% training

def TRAIN(net, optimizer, X_train, Y_train, X_test, Y_test, epochs=3000, loss_fn=nn.CrossEntropyLoss()):
    train_loss_log = []
    test_loss_log = []
    #X_train = torch.from_numpy(X_train).to(device)
    #Y_train = torch.from_numpy(Y_train).to(device)
    #X_test  = torch.from_numpy(X_test).to(device)
    #Y_test  = torch.from_numpy(Y_test).to(device)
    for ep in trange(epochs):
        net.train()
        optimizer.zero_grad()
        out = net(X_train)
        train_loss = loss_fn(out, Y_train)
        train_loss.backward()
        optimizer.step()
        # Validation
        net.eval() # Evaluation mode (e.g. disable dropout)
        with torch.no_grad(): # No need to track the gradients
            out = net(X_test)
            # Evaluate global loss
            test_loss = loss_fn(out, Y_test)
        train_loss_log.append(float(train_loss))
        test_loss_log .append(float(test_loss))
        """if (ep+1)%100 == 0:
            print(' '*5+"Epoche {} - Train Loss = {:.4f} - Test Loss = {:.4f}".format(ep+1, float(train_loss), float(test_loss)))"""
    return net, train_loss_log, test_loss_log

# %% Kfold cross validation

def kfoldCV(net, optimizer, X, Y, nfolds=5, epochs=3000, loss_fn=nn.CrossEntropyLoss(), rs=100):
    kf = KFold(nfolds, shuffle=True, random_state=rs)
    current_Fold = 1
    validation_losses = []
    for train_index, test_index in kf.split(X):
        print("CV split {} / {}".format(current_Fold, nfolds))

        # get train/validation folds
        X_train, X_val = X[train_index], X[test_index]
        Y_train, Y_val = Y[train_index], Y[test_index]
        _, _, val_loss_log = TRAIN(net, optimizer, X_train, Y_train, X_val, Y_val, epochs=epochs, loss_fn=loss_fn)
        validation_losses.append(val_loss_log[-1])

        current_Fold += 1

    validation_loss = np.mean(validation_losses)

    return validation_loss

# %% parameters search
def ParametersSearch(Nh1, Nh2, optimizer, X, Y, nfolds=5, epoches=3000, device="cpu"):
    results = pd.DataFrame(columns=["Nh1", "Nh2", "Optimizer", "Val_Loss"])
    
    total_cycles = len(Nh1)*len(Nh2)*len(optimizer)
    it = 1
    
    for n1 in Nh1:
        for n2 in Nh2:
            for opt_name, opt in optimizer.items():
                print("Nh1 = {} | Nh2 = {} | Opt = {} | Iter {} / {}".format(n1,n2,opt_name, it, total_cycles))
                net = Net(Nh1=n1, Nh2=n2)
                net.to(device)
                optim = opt(net.parameters())
                val_loss = kfoldCV(net, optim, X, Y, nfolds=nfolds, epochs=epoches, rs=None)
                log = pd.DataFrame({
                    "Nh1"       : [n1],
                    "Nh2"       : [n2],
                    "Optimizer" : [opt_name],
                    "Val_Loss"  : [val_loss]
                })
                results = results.append(log)
                it += 1
    results.sort_values(by="Val_Loss", inplace=True)
    return results

# %% Random Search
np.random.seed(784012)
Ni = 784
Nh1 = [256] + list(np.random.randint(100, 500, size=9))
Nh2 = [256] + list(np.random.randint(100, 500, size=9))
No = 10
optimizer = {"Adam":torch.optim.Adam, "RMSprop":torch.optim.RMSprop}

results = ParametersSearch(Nh1, Nh2, optimizer, X_train, Y_train, device=device, epoches=1000)

# %% save results dataframe and get best model
print("Parameters search results:\n", results, "\n\n")
results.to_csv("GridSearch_csv.csv"    , index=False)
results.to_latex("GridSearch_latex.txt", index=False)

best = results.iloc[0].to_dict()
print("Best Model:\n", best)
print("Validation loss =", best["Val_Loss"], "\n\n")
f = open("best_model.txt", "w")
print("Best Model:\n", best, file=f)
f.close()

# %% train best model on the whole training set and save it
net = Net(Nh1=best["Nh1"], Nh2=best["Nh2"])
net.to(device)
opt = optimizer[best["Optimizer"]]
opt = optim = opt(net.parameters())
net, train_loss_log, test_loss_log = TRAIN(net, opt, X_train, Y_train, X_test, Y_test, epochs=3000, loss_fn=nn.CrossEntropyLoss())
torch.save(net.state_dict(), "best_model.torch")

# %% plot losses
plt.close("all")
plt.figure(figsize=(8,5))
plt.semilogy(train_loss_log, label='Train loss')
plt.semilogy(test_loss_log,  label='Test loss')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.grid()
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig("Loss.pdf")
#plt.show()

# %% prediction on test set
net.eval()
with torch.no_grad():
    y_pred = net(X_test)
y_pred = F.log_softmax(y_pred, dim=1)
_, y_class = y_pred.max(1)
np_Y_test = Y_test.cpu().numpy()
np_y_pred = y_class.cpu().numpy()
correct = np.sum(np_Y_test == np_y_pred)
print("Accuracy = {:.5f} %".format(correct*100/np_Y_test.shape[0]))
f = open("best_model.txt", "a")
f.write("\n\nAccuracy on test set = {:.5f} %".format(correct*100/np_Y_test.shape[0]))
f.close()


# %% Access network parameters
h1_w = net.fc1.weight.cpu().data.numpy()
h1_b = net.fc1.bias.cpu().data.numpy()
h2_w = net.fc2.weight.cpu().data.numpy()
h2_b = net.fc2.bias.cpu().data.numpy()
h3_w = net.fc3.weight.cpu().data.numpy()
h3_b = net.fc3.bias.cpu().data.numpy()

# %% Plot last layer receptive field
"""plt.figure(figsize=(8,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.axis("off")

    reshaped =
    plt.imshow(h3_w[i].reshape(16,16).T, cmap="gray")
    plt.title(i)
plt.show()"""

# %% Weights histogram
plt.close("all")
fig, axs = plt.subplots(3, 1, figsize=(12,8))
axs[0].hist(h1_w.flatten(), 50)
axs[0].set_title('First hidden layer weights')
axs[1].hist(h2_w.flatten(), 50)
axs[1].set_title('Second hidden layer weights')
axs[2].hist(h3_w.flatten(), 50)
axs[2].set_title('Output layer weights')
[ax.grid() for ax in axs]
plt.tight_layout()
plt.savefig("Weights.pdf")
#plt.show()




"""#%% Training
train_loss_log = []
test_loss_log = []
num_epochs = 3000
for ep in range(num_epochs):
    # Training
    net.train() # Training mode (e.g. enable dropout)
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    #X_train.to(device)
    #Y_train.to(device)
    out = net(X_train)
    # Evaluate loss
    loss = loss_fn(out, Y_train)
    # Backward pass
    loss.backward()
    # Update
    optimizer.step()

    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        out = net(X_test)
        # Evaluate global loss
        test_loss = loss_fn(out, Y_test)

    # Print loss
    print("Epoche {} - Train Loss = {:.4f} - Test Loss = {:.4f}".format(ep, float(loss), float(test_loss)))

    # Log
    train_loss_log.append(float(loss.data))
    test_loss_log.append(float(test_loss.data))"""

"""# %% Plot losses
plt.close('all')
plt.figure(figsize=(12,8))
plt.semilogy(train_loss_log, label='Train loss')
plt.semilogy(test_loss_log, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()"""


"""# %% prediction on test set
with torch.no_grad():
    y_pred = net(X_test)
y_pred = F.log_softmax(y_pred, dim=1)
_, y_class = y_pred.max(1)
np_Y_test = Y_test.cpu().numpy()
np_y_pred = y_class.cpu().numpy()
correct = np.sum(np_Y_test == np_y_pred)
print("Accuracy = {:.5f} %".format(correct*100/np_Y_test.shape[0]))"""

"""# %% Print some examples
# take 4 random images
np.random.seed(24650)
idx = np.random.randint(0, X_test.shape[0]+1, 4)
plt.figure()
for i,img in enumerate(idx):
    plt.subplot(2,2,i+1)
    plt.axis("off")
    plt.imshow(X_test[img].cpu().reshape(28,28).T, cmap="gray")
    plt.title("Real {} - Predict {}".format(np_Y_test[img], np_y_pred[img]))"""


"""# %% Access network parameters
h1_w = net.fc1.weight.cpu().data.numpy()
h1_b = net.fc1.bias.cpu().data.numpy()
h2_w = net.fc2.weight.cpu().data.numpy()
h2_b = net.fc2.bias.cpu().data.numpy()
h3_w = net.fc3.weight.cpu().data.numpy()
h3_b = net.fc3.bias.cpu().data.numpy()"""

"""# %% Plot last layer receptive field
plt.figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.axis("off")
    plt.imshow(h3_w[i].reshape(16,16).T, cmap="gray")
    plt.title(i)
"""
# %%
