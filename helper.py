import random
import copy
from collections import defaultdict

import numpy as np
import scipy
from scipy.stats import trim_mean
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models

from FmnistCnn import FmnistCnn
from ResNet import *


# Load training and test datasets
def load_data(data="fmnist", normalize=True):
    # Load FMNIST dataset
    if data == "fmnist":
        if normalize:
            transform_train = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
        else:
            transform_train = transforms.ToTensor()
            transform_test = transforms.ToTensor()

        train_dataset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True, transform=transform_train)
        test_dataset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False, transform=transform_test)

    # Load CIFAR10 dataset
    if data == "cifar10":
        if normalize:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.ToTensor()
            transform_test = transforms.ToTensor()

        train_dataset = datasets.CIFAR10('CIFAR10_data/', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('CIFAR10_data/', train=False, transform=transform_test)

    return(train_dataset, test_dataset)

# Returns the local data ratio and dataloader per user
def distribute_data(train_dataset, batch_size, n_users, iid_flag=True):
    all_data_size = len(train_dataset)
    # 1) iid distribution: allocate data of uniform size to users
    if iid_flag:
        data_range = list(range(len(train_dataset)))
        random.shuffle(data_range)
        data_size = int(len(train_dataset)/n_users)
        user_data = {i: [data_size/all_data_size, torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                data_range[i*data_size:(i+1)*data_size]))]
                     for i in range(n_users)}

    # 2) non-iid distribution: allocate data per Dirichlet distribution to users
    else:
        alpha = 0.9
        # Store the indices of images corresponding to each class label
        img_label_inds = defaultdict(list)
        for ind, data in enumerate(train_dataset):
            img_label_inds[data[1]].append(ind)
        n_labels = len(img_label_inds)
        # Allocate images per dirichelt distribution for each class label
        user_data_tmp = defaultdict(list)
        for label in range(n_labels):
            random.shuffle(img_label_inds[label])
            n_imgs = len(img_label_inds[label])
            n_samples_per_user = n_imgs * np.random.dirichlet(n_users * [alpha])
            for user in range(n_users):
                n_samples = int(round(n_samples_per_user[user]))
                sample_list = img_label_inds[label][:n_samples]
                user_data_tmp[user].extend(sample_list)
                img_label_inds[label] = img_label_inds[label][n_samples:]
        user_data = {i: [len(user_data_tmp[i])/all_data_size,
                         torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                user_data_tmp[i]))]
                     for i in range(n_users)}

    return(user_data)

# Create models corresponding to the dataset
def create_model(data="fmnist"):
    if data == "fmnist":
        global_model = FmnistCnn()
    if data == "cifar10":
        global_model = ResNet18()
    global_model.cuda()
    return(global_model)

# Reports the average loss and accuracy of the model on test dataset
def test(model, test_dataset):
    def get_testdata_loader(test_dataset):
        test_batch_size = 20
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
        return test_loader

    with torch.no_grad():
        dataset_size = len(test_dataset)
        test_dataloader = get_testdata_loader(test_dataset)
        model.eval()
        total_loss = 0
        correct = 0
        for batch_id, batch in enumerate(test_dataloader):
            images, labels = batch
            images, labels = images.cuda(), labels.cuda()
            images.requires_grad_(False)
            labels.requires_grad_(False)
            output = model(images)
            total_loss += nn.functional.cross_entropy(output, labels, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()

        accuracy = correct/dataset_size
        avg_loss = total_loss/dataset_size
    model.train()
    return(accuracy, avg_loss)

# Train one local model with SGD and save the new weight and update parameters at one epoch
def sgd_train_one_local_model(current_user, global_model, users_data, local_gradient,
                          weighed_local_weight, local_iterations=500, byzantine=False, weighted=True):
    # Local training configuraitons (to be class params)
    lr = 0.01
    momentum = 0.9

    current_user_data = users_data[current_user][1]
    # Initialize the local parameters with the global paramameters
    local_model = copy.deepcopy(global_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)

    # Train the local model
    local_model.train()
    epoch = 0
    while epoch < local_iterations:
        for i, data in enumerate(current_user_data):
            optimizer.zero_grad()
            images, labels = data
            if byzantine:
                labels = 9-labels
            images, labels = images.cuda(), labels.cuda()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch = epoch + 1
            if epoch >= local_iterations:
                break

    # Store the weight delta of each edge user
    for name, data in local_model.state_dict().items():
        weight_delta = global_model.state_dict()[name] - data
        local_gradient[name][current_user] = weight_delta
        if weighted:
            current_p = users_data[current_user][0]
            weighed_local_weight[name][current_user] = data*current_p
        else:
            weighed_local_weight[name][current_user] = data
    return(local_gradient, weighed_local_weight)

# Compute one local model's loss gradient and save the new weight and update parameters at one epoch
def oneshot_train_one_local_model(current_user, global_model, users_data_iterator,
                                    local_gradient, local_weight, byzantine=False):

    current_user_data = users_data_iterator[current_user]
    # Initialize the local parameters with the global paramameters
    local_model = copy.deepcopy(global_model)
    local_model.train()
    criterion = nn.CrossEntropyLoss()

    # Compte the local gradient
    images, labels = next(current_user_data)
    if byzantine:
        labels = 9-labels
    images = images.cuda(); labels = labels.cuda()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=1)
    optimizer.zero_grad()
    outputs = local_model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    for name, data in local_model.state_dict().items():
        weight_delta = global_model.state_dict()[name] - data
        local_gradient[name][current_user] = weight_delta
        local_weight[name][current_user] = data

    return(local_gradient, local_weight)

# Aggregate the local updates with FedAvg algorithm
def federated_average_aggregate(global_model, local_weight, n_users, n_chosen_models, weighted=False):
    multiplier = n_users/n_chosen_models
    for name, _ in global_model.state_dict().items():
        new_data = multiplier * sum(local_weight[name].values())  # weighted sum of local params
        if weighted:
            weight = 1/n_users
            global_model.state_dict()[name].copy_(weight * new_data)
        else:
            global_model.state_dict()[name].copy_(new_data)
    return(global_model)
