# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
from torch import optim, nn
from dataset import LoadDataset, ToTensor
from network import Network, train_batch, test_batch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from pathlib import Path
from shutil import copytree, rmtree
from os.path import isdir

def plot_loss(training_loss, test_loss, fname=None):

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(train_loss, label='Train loss')
    ax.plot(test_loss,  label='Test loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.grid()
    ax.legend(fontsize=14)
    fig.tight_layout()
    if fname is not None: fig.savefig(fname)
    #plt.show()

    return


if __name__ == '__main__':

    args = {
        "datasetpath"   : "Scienza_in_cucina.txt",
        "crop_len"      : 15,
        "embedding_dim" : 100,
        "input_size"    : None,
        "hidden_units"  : None,
        "layers_num"    : None,
        "dropout_prob"  : 0.3,
        "batchsize"     : 500,
        "num_epochs"    : 350,
        "out_dir"       : None
    }

    # set seed
    np.random.seed(43789235)

    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)
    #%% Create dataset and split into train and test
    trans = ToTensor()
    dataset = LoadDataset(filepath=args["datasetpath"], crop_len=args["crop_len"], transform=trans, embedding=args["embedding_dim"])
    args["input_size"] = dataset.vocabulary_length()
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print("train = {} | test = {}".format(train_size, test_size))
    # Define Dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args["batchsize"], shuffle=True, num_workers=1)
    test_dataloader  = DataLoader( test_dataset, batch_size=args["batchsize"], shuffle=True, num_workers=1)

    # hyperparameters search

    hidden_units = [64, 128, 256]
    layers_num   = [2, 3, 4]

    cnt = 0

    results = pd.DataFrame(columns=["Hidden_Units", "Layers_Num", "Out_Dir", "Train_Loss", "Test_Loss"])

    for hu in hidden_units:
        for ln in layers_num:
            cnt += 1
            args["hidden_units"] = hu
            args["layers_num"]   = ln
            args["out_dir"]      = "Model{:02d}".format(cnt)

            print("="*50)
            print("Combination {:2d} / {:2d}".format(cnt, len(hidden_units)*len(layers_num)))
            print("hidden_units = {} | layers_num = {} | out_dir = {}".format(hu, ln, args["out_dir"]))
            print("="*50)

            # Create output dir
            out_dir = Path(args["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save training parameters
            with open(out_dir / 'training_args.json', 'w') as f:
                json.dump(args, f, indent=4)
            # Save encoder dictionary
            with open(out_dir / 'char_to_number.json', 'w') as f:
                json.dump(dataset.word2idx, f, indent=4)
            # Save decoder dictionary
            with open(out_dir / 'number_to_char.json', 'w') as f:
                json.dump(dataset.idx2word, f, indent=4)

            #%% Initialize network
            net = Network(input_size  =args["input_size"],
                          hidden_units=args["hidden_units"],
                          layers_num  =args["layers_num"],
                          embedding   =args["embedding_dim"],
                          dropout_prob=args["dropout_prob"])
            net.to(device)

            #%% Train network

            # Define optimizer
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), weight_decay=5e-4)
            scheduler = ReduceLROnPlateau(optimizer, factor=0.75, patience=10, verbose=False, threshold=1e-04, cooldown=5, min_lr=1e-07)
            # Define loss function
            loss_fn = nn.CrossEntropyLoss()

            # loss logs
            train_loss, test_loss = [], []

            # early stopping
            early_stop = 15
            n_ep_stop = 0

            # Start training
            for epoch in range(args["num_epochs"]):

                train_l = train_batch(net, train_dataloader, loss_fn, optimizer, device)
                test_l  = test_batch (net,  test_dataloader, loss_fn, optimizer, device)
                # save losses
                train_loss.append(train_l)
                test_loss .append(test_l)

                # update optimizer scheduler
                scheduler.step(test_l)

                # early stop if no improvement in early_stop consecutive iterations
                if epoch > 0 and test_loss[-1] >= test_loss[-2]:
                    n_ep_stop += 1
                    if n_ep_stop == early_stop:
                        print("#"*10)
                        print("Early stop at epoch {}:".format(epoch))
                        print("No improvement in test loss during last {} iterations".format(early_stop))
                        print("#"*10)
                        break
                else:
                    n_ep_stop = 0

                # print information about training ans save network status
                if ((epoch+1) % 50 == 0):
                    print("{}ep = {:5d} | train = {:7.4f} | test = {:7.4f}". format(" "*5, epoch+1, train_l, test_l))
                    # Save network parameters
                    torch.save(net.state_dict(), out_dir / 'net_params.pth')

            # Save final network parameters
            torch.save(net.state_dict(), out_dir / 'net_params.pth')
            plot_loss(train_loss, test_loss,  out_dir / 'Loss.pdf')
            np.savez_compressed(out_dir / 'loss.npz', train=train_loss, test=test_loss)

            log = pd.DataFrame({
                "Hidden_Units" : [hu],
                "Layers_Num"   : [ln],
                "Out_Dir"      : args["out_dir"],
                "Train_Loss"   : [train_loss[-1]],
                "Test_Loss"    : [test_loss[-1]]

            })
            results = results.append(log)

    results.sort_values(by="Test_Loss", inplace=True)

    # %% save results dataframe and get best model
    print("Parameters search results:\n", results, "\n\n")
    results.to_csv("ParamSearch_csv.csv"    , index=False)
    results.to_latex("ParamSearch_latex.txt", index=False)

    best = results.iloc[0].to_dict()
    with open('best_model.json', 'w') as f:
        json.dump(best, f, indent=4)

    # copy best model to a specific folder
    folder = Path(best["Out_Dir"])
    out_dir = Path("BestModel")
    if isdir(out_dir): rmtree(out_dir)
    copytree(folder, out_dir)
