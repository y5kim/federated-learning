import random
import copy
from collections import defaultdict
import pickle
import copy
import itertools as it

import numpy as np
import scipy
from scipy.stats import trim_mean
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

from helper import *


def dlg_one_batch(imgs, labels, globa_model, n_iters, save=True):

    # Convert labels to one hot vectors
    def label_to_onehot(labels, n_class=10):
        labels = torch.unsqueeze(labels, 1)
        onehot_labels = torch.zeros(labels.size()[0], n_class).cuda()
        onehot_labels.scatter_(1, labels, 1)
        return onehot_labels

    # Compute the average cross entorpy given labels and predictions
    def mean_cross_entropy(labels, preds):
        tmp = torch.sum(- labels * F.log_softmax(preds, dim=-1), 1)
        return torch.mean(tmp)

    n_classes = 10
    local_model = copy.deepcopy(global_model)

    imgs = imgs.cuda(); labels = labels.cuda()
    labels = label_to_onehot(labels, n_classes)
    criterion = mean_cross_entropy

    # Compute the original local graident
    preds = local_model(imgs)
    loss = criterion(labels, preds)
    orig_grad = torch.autograd.grad(loss, local_model.parameters())
    orig_grad = [x.detach().clone() for x in orig_grad]  # detach the gradients

    dummy_data_list = []
    dummy_labels_list = []
    loss_list = []

    # Initialize the dummy data and label
    dummy_data = torch.randn(imgs.size()).cuda().requires_grad_(True)
    dummy_onehot_labels = torch.randn(labels.size()[0], n_classes).cuda().requires_grad_(True)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_onehot_labels])
    # Update the dummy data iteratively by minimizing the distance beween original and dummy gradients
    for i in range(n_iters):
        def closure():
            optimizer.zero_grad()
            dummy_preds = local_model(dummy_data)
            dummy_softmax_labels = F.softmax(dummy_onehot_labels, dim=-1)
            dummy_loss = criterion(dummy_softmax_labels, dummy_preds)
            dummy_grad = torch.autograd.grad(dummy_loss, local_model.parameters(), create_graph=True)
            grad_diff = 0
            for dum, org in zip(dummy_grad, orig_grad):
                grad_diff += ((dum-org)**2).sum()
            grad_diff.backward()
            return(grad_diff)
        optimizer.step(closure)
        checkpoint = 10
        if i % checkpoint == 0:
            current_loss = closure()
            print(current_loss.item())
            dummy_data_list.append(copy.deepcopy(dummy_data))
            dummy_labels_list.append(copy.deepcopy(dummy_onehot_labels))
            loss_list.append(current_loss.item())

    # Save outputs
    if save:
        with open('image_outputs/{}_orig_imgs_{}.pickle'.format(dataset, batch_size), 'wb') as handle:
            pickle.dump(imgs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('image_outputs/{}_orig_labels_{}.pickle'.format(dataset, batch_size), 'wb') as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('image_outputs/{}_rec_imgs_{}.pickle'.format(dataset, batch_size), 'wb') as handle:
            pickle.dump(dummy_data_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('image_outputs/{}_rec_labels_{}.pickle'.format(dataset, batch_size), 'wb') as handle:
            pickle.dump(dummy_labels_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return(imgs, labels, dummy_data_list, dummy_labels_list, loss_list)

def compare_images(imgs, labels, dummy_data_list, dummy_labels_list):
    batch_size = imgs.shape[0]
    to_pil = transforms.ToPILImage()
    dummy_data = dummy_data_list[-1]
    dummy_onehot_labels = dummy_labels_list[-1]
    # Original images
    max_labels = torch.argmax(labels, -1)
    fig1 = plt.figure(figsize=(12, 8))
    for i in range(batch_size):
        plt.subplot(batch_size//4 + 1, 4, i + 1)
        plt.imshow(to_pil(imgs[i].cpu()))
        plt.title("label: {}".format(max_labels[i]))
        plt.axis('off')

    # Recovered images
    max_preds = torch.argmax(dummy_onehot_labels,-1)
    fig2 = plt.figure(figsize=(12, 8))
    for i in range(batch_size):
        plt.subplot(batch_size//4 + 1, 4, i + 1)
        tmp = dummy_data.detach().cpu()[i]
        minv = torch.min(tmp)
        maxv = torch.max(tmp)
        normed = ((tmp - minv) / (maxv - minv))
        plt.imshow(to_pil(normed))
        plt.title("label: {}".format(max_preds[i]))
        plt.axis('off')

    plt.show()
    return(fig1, fig2)

def display_progressions(dummy_data_list, img, ind=0):
    to_pil = transforms.ToPILImage()
    dummy_data_list = dummy_data_list
    n_imgs = len(dummy_data_list)
    checkpoint = 10
    fig = plt.figure(figsize=(12, 8))
    for i in range(1,n_imgs+1,2):
        plt.subplot(n_imgs//6 + 1, 6, i//2 + 1)
        tmp = dummy_data_list[i].detach().cpu()[ind]
        minv = torch.min(tmp)
        maxv = torch.max(tmp)
        normed = ((tmp - minv) / (maxv - minv))
        plt.imshow(to_pil(normed))
        plt.title("Iterations: {}".format(checkpoint*(i+1)))
        plt.axis('off')
    plt.subplot(n_imgs//6 + 1, 6, 6)
    plt.imshow(to_pil(img.cpu()))
    plt.title("Original image")
    plt.axis('off')
    plt.show()
    return(fig)


if __name__ == '__main__':
    dataset = "cifar10"  #"fmnist"
    train_dataset, test_dataset = load_data(dataset, False)
    global_model = create_model(dataset)
    img_label_inds = defaultdict(list)
    for ind, data in enumerate(train_dataset):
        img_label_inds[data[1]].append(ind)

    for k in range(10):
        for j in np.arange(0,10):
            i1 = 4
            j1 = img_label_inds[j][k+1]
            dist = torch.mean(abs(train_dataset[i1][0] - train_dataset[j1][0]))
            imgs = torch.stack((train_dataset[i1][0], train_dataset[j1][0]))
            labels = torch.stack(( torch.tensor(train_dataset[i1][1]), torch.tensor(train_dataset[j1][1]) ))
            n_iters = 100
            imgs, labels, dummy_data_list, dummy_label_list, loss_list = dlg_one_batch(imgs, labels, global_model,
                                                                            n_iters, save=False)
            # fig1, fig2 = compare_images(imgs, labels, dummy_data_list, dummy_label_list)
