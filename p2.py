import random
import copy
from collections import defaultdict
import pickle

import numpy as np
import scipy
from scipy.stats import trim_mean
import torch
import torch.utils.data
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models

from helper import *
from FmnistCnn import FmnistCnn
from ResNet import *


# Train local models for one epoch in federated learning
def train_epoch(global_model, users_data, n_users, n_chosen_models, local_iterations=500):
    # Initialize the weight accumulator
    local_gradient = defaultdict(lambda: defaultdict(dict))
    weighed_local_weight = defaultdict(lambda: defaultdict(dict))

    # Select models for the epoch
    selected_models = np.random.choice(n_users, n_chosen_models, False)
    # Train each local model
    for current_user in selected_models:
        local_gradient, weighed_local_weight = sgd_train_one_local_model(current_user, global_model, users_data,
                                                                     local_gradient, weighed_local_weight, local_iterations)
    return(local_gradient, weighed_local_weight)

# Run federated learning
def run_federated_learning(dataset, total_epochs, batch_size, n_users, n_chosen_models, iid_flag=True):
    loss_list = []; acc_list = []
    # Initalize data and model
    train_dataset, test_dataset = load_data(dataset)
    users_data = distribute_data(train_dataset, batch_size, n_users, iid_flag)
    global_model = create_model(dataset)

    accuracy, avg_loss = test(global_model, test_dataset)
    acc_list.append(accuracy); loss_list.append(avg_loss)
    print("Epoch {}: accuracy {}, avg loss {}".format(0, accuracy, avg_loss))

    for epoch in range(total_epochs):
        # Train local models and aggregate the parameters
        local_gradient, weighed_local_weight = train_epoch(global_model, users_data, n_users, n_chosen_models)
        global_model = federated_average_aggregate(global_model, weighed_local_weight, n_users, n_chosen_models)
        accuracy, avg_loss = test(global_model, test_dataset)
        acc_list.append(accuracy); loss_list.append(avg_loss)
        print("Epoch {}: accuracy {}, avg loss {}".format(epoch+1, accuracy, avg_loss))

    return(global_model, loss_list, acc_list)

# Run centralized learning on a given portion of training dataset
def run_centralized_learning(dataset, total_epochs, batch_size, train_data_portion=1):
    lr = 0.01
    momentum = 0.9

    loss_list = []; acc_list = []
    # Initalize data and model
    train_dataset, test_dataset = load_data(dataset)
    global_model = create_model(dataset)
    train_data_size = int(len(train_dataset)*train_data_portion)
    data_range = list(range(len(train_dataset)))
    accuracy, avg_loss = test(global_model, test_dataset)
    acc_list.append(accuracy); loss_list.append(avg_loss)
    print("Epoch {}: accuracy {}, avg loss {}".format(0, accuracy, avg_loss))

    # Train the global model over epochs
    for epoch in range(total_epochs):
        random.shuffle(data_range)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       data_range[:train_data_size]))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(global_model.parameters(), lr=lr, momentum=momentum)
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        accuracy, avg_loss = test(global_model, test_dataset)
        acc_list.append(accuracy); loss_list.append(avg_loss)
        print("Epoch {}: accuracy {}, avg loss {}".format(epoch+1, accuracy, avg_loss))

    return(global_model, loss_list, acc_list)

# Run full experiment of CL and FL and save the outputs
def run_e2e_experiment(dataset, n_users_list, iid_flag_list, total_epochs, reload=True):
    # Default configs
    n_chosen_models = 10; batch_size = 20;
    # Reload if there are saved outputs
    if reload:
        with open('outputs/{}_cl_loss.pickle'.format(dataset), 'rb') as handle:
            cl_loss = pickle.load(handle)

        with open('outputs/{}_cl_acc.pickle'.format(dataset), 'rb') as handle:
            cl_acc = pickle.load(handle)

        with open('outputs/{}_fl_loss.pickle'.format(dataset), 'rb') as handle:
            fl_loss = pickle.load(handle)

        with open('outputs/{}_fl_acc.pickle'.format(dataset), 'rb') as handle:
            fl_acc = pickle.load(handle)
    else:
        cl_loss = {}; cl_acc = {}
        fl_loss = {}; fl_acc = {}

    # Run centralized learning
    central_model, central_loss, central_acc = run_centralized_learning(dataset, total_epochs, batch_size, 1)
    cl_loss[n_chosen_models] = central_loss; cl_acc[n_chosen_models] = central_acc

    with open('outputs/{}_cl_loss_new.pickle'.format(dataset), 'wb') as handle:
        pickle.dump(cl_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('outputs/{}_cl_acc_new.pickle'.format(dataset), 'wb') as handle:
        pickle.dump(cl_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Run federated learning
    for n_users in n_users_list:
        print("#### Total number of users: {} ####".format(n_users))
        fl_loss[n_users] = {}; fl_acc[n_users] = {}
        for iid_flag in iid_flag_list:
            print("###### IID data distribution: {} ######".format(iid_flag))
            fed_model, fed_loss, fed_acc = run_federated_learning(dataset, total_epochs, batch_size, n_users,
                                                                  n_chosen_models, iid_flag)
            fl_loss[n_users][iid_flag] = fed_loss; fl_acc[n_users][iid_flag] = fed_acc

            with open('outputs/{}_fl_loss_{}.pickle'.format(dataset, total_epochs), 'wb') as handle:
                pickle.dump(fl_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('outputs/{}_fl_acc_{}.pickle'.format(dataset, total_epochs), 'wb') as handle:
                pickle.dump(fl_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return(cl_loss, cl_acc, fl_loss, fl_acc)


if __name__ == '__main__':
    dataset = "fmnist"
    n_users_list = [100, 20, 10]
    iid_flag_list = [True, False]
    total_epochs = 20

    cl_loss, cl_acc, fl_loss, fl_acc = run_e2e_experiment(dataset, n_users_list,
        iid_flag_list, total_epochs, False)
