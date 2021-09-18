import random
import copy
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import scipy
from scipy.stats import trim_mean
import torch
import torch.utils.data
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

from helper import *


## Aggregation methods
def median_aggregate_updates(global_model, local_gradient, lr):
    new_global_model = copy.deepcopy(global_model)
    for name, data in global_model.state_dict().items():
        local_vals = list(local_gradient[name].values())
        stacked_local_vals = torch.stack(local_vals)
        median_vals = torch.median(stacked_local_vals, dim=0)[0]
        new_data = data - lr*median_vals
        new_global_model.state_dict()[name].copy_(new_data)
    return(new_global_model)

def median_aggregate_weights(global_model, local_weight):
    new_global_model = copy.deepcopy(global_model)
    for name, data in global_model.state_dict().items():
        local_vals = list(local_weight[name].values())
        stacked_local_vals = torch.stack(local_vals)
        median_vals = torch.median(stacked_local_vals, dim=0)[0]
        new_global_model.state_dict()[name].copy_(median_vals)
    return(new_global_model)

def trimmed_mean_aggregate_updates(global_model, local_gradient, lr, beta=0.1):
    new_global_model = copy.deepcopy(global_model)
    for name, data in global_model.state_dict().items():
        local_vals = list(local_gradient[name].values())
        stacked_local_vals = torch.stack(local_vals)
        stacked_array = stacked_local_vals.cpu().detach().numpy()
        trimmed_mean_array = trim_mean(stacked_array, beta/2)
        if data.shape == tuple():
            trimmed_mean_array = np.array(trimmed_mean_array)
        trimmed_mean_vals = torch.from_numpy(trimmed_mean_array).float().cuda()
        new_data = data - lr*trimmed_mean_vals
        new_global_model.state_dict()[name].copy_(new_data)
    return(new_global_model)

# Run one-round Byzantine robust FL algorithm
def run_oneshot_byzantine_federated_learning(dataset, n_users, local_iters, byzantine_ratio=0):

    loss_list_avg = []; acc_list_avg = []
    loss_list_med = []; acc_list_med = []
    local_gradient = defaultdict(lambda: defaultdict(dict))
    local_weight = defaultdict(lambda: defaultdict(dict))
    # Initalize data and model
    train_dataset, test_dataset = load_data(dataset)
    batch_size = int(len(train_dataset)/n_users)
    users_data = distribute_data(train_dataset, batch_size, n_users)
    global_model = create_model(dataset)

    byzantine_models = random.sample(list(np.arange(n_users)), int(byzantine_ratio*n_users))
    print("Selected Byzantine models: ", byzantine_models)

    accuracy, avg_loss = test(global_model, test_dataset)
    acc_list_avg.append(accuracy); loss_list_avg.append(avg_loss)
    acc_list_med.append(accuracy); loss_list_med.append(avg_loss)
    print("Epoch {}: accuracy {}, avg loss {}".format(0, accuracy, avg_loss))

    # Train all the local models
    for current_user in range(n_users):
        byzantine = (current_user in byzantine_models)
        local_gradient, local_weight = sgd_train_one_local_model(current_user, global_model, users_data,
                                                                 local_gradient, local_weight,
                                                                 local_iters, byzantine, False)

    global_model_avg = federated_average_aggregate(global_model, local_weight, n_users, n_users, True)
    global_model_med = median_aggregate_weights(global_model, local_weight)

    accuracy_avg, avg_loss_avg = test(global_model_avg, test_dataset)
    print("Epoch {}: baseline accuracy {}, baseline avg loss {}".format(1, accuracy_avg, avg_loss_avg))
    accuracy_med, avg_loss_med = test(global_model_med, test_dataset)
    print("Epoch {}: byzantine-robust accuracy {}, byzantine-robust avg loss {}".format(1, accuracy_med, avg_loss_med))
    acc_list_avg.append(accuracy_avg); loss_list_avg.append(avg_loss_avg)
    acc_list_med.append(accuracy_med); loss_list_med.append(avg_loss_med)

    return(acc_list_avg, loss_list_avg, acc_list_med, loss_list_med)

# Run gradient descent Byzantine robust FL algorithm
def run_gd_byzantine_federated_learning(dataset, total_epochs, n_users, agg_method="avg", byzantine_ratio=0):
    loss_list = []; acc_list = []
    local_gradient = defaultdict(lambda: defaultdict(dict))
    local_weight = defaultdict(lambda: defaultdict(dict))
    # Initalize data and model
    train_dataset, test_dataset = load_data(dataset)
    batch_size = int((len(train_dataset)/n_users)*0.1)
    users_data = distribute_data(train_dataset, batch_size, n_users)
    global_model = create_model(dataset)

    byzantine_models = random.sample(list(np.arange(n_users)), int(byzantine_ratio*n_users))
    print("Selected Byzantine models: ", byzantine_models)

    accuracy, avg_loss = test(global_model, test_dataset)
    acc_list.append(accuracy); loss_list.append(avg_loss)
    print("Epoch {}: accuracy {}, avg loss {}".format(0, accuracy, avg_loss))

    for epoch in range(total_epochs):
        if epoch%10 == 0:
            users_data_iterator = {user: iter(users_data[user][1]) for user in range(n_users)}
        # Train all the local models
        for current_user in range(n_users):
            byzantine = (current_user in byzantine_models)
            local_gradient, local_weight = oneshot_train_one_local_model(current_user, global_model, users_data_iterator,
                                                   local_gradient, local_weight, byzantine)
        if agg_method == "avg":
            global_model = federated_average_aggregate(global_model, local_weight, n_users, n_users, True)
        elif agg_method == "median":
            global_model = median_aggregate_updates(global_model, local_gradient, 0.05)
        elif agg_method == "trim_mean":
            global_model = trimmed_mean_aggregate_updates(global_model, local_gradient, 0.05, beta=byzantine_ratio)
        accuracy, avg_loss = test(global_model, test_dataset)
        acc_list.append(accuracy); loss_list.append(avg_loss)
        if epoch % 50 == 0:
            print("Epoch {}: accuracy {}, avg loss {}".format(epoch+1, accuracy, avg_loss))

    return(global_model, loss_list, acc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("alg", help="choose which algorithm to run from oneround and gradient descent")
    args = parser.parse_args()

    if args.alg == "oneshot":
        # One-shot learnign
        dataset="fmnist"; n_users=20; local_iters=500
        byzantine_ratio_list = [0, 0.1, 0.2, 0.4]

        for alpha in byzantine_ratio_list:
            print("Byzantine ratio: {}".format(alpha))
            acc_list_avg, loss_list_avg, acc_list_med, loss_list_med = run_oneshot_byzantine_federated_learning(dataset, n_users, local_iters, alpha)
            df = pd.DataFrame(list(zip(acc_list_avg, loss_list_avg, acc_list_med, loss_list_med)),
            columns =['avg_accuracy', 'avg_loss', 'median_accuracy', "median_loss"])
            df.to_csv("{}/p3_oneshot_{}_{}_{}_{}.csv".format("outputs", dataset, alpha, n_users, local_iters))

    if args.alg == "gd":
        dataset = "fmnist"; total_epochs = 500; n_users = 20
        byzantine_ratio_list = [0, 0.1, 0.2, 0.4]; agg_list = ["median", "trim_mean", "avg"]

        for alpha in byzantine_ratio_list:
            print("Byzantine ratio: {}".format(alpha))
            acc_dict = {}; loss_dict = {}
            for agg in agg_list:
                print("Aggregation method: {}".format(agg))
                _, loss, acc = run_gd_byzantine_federated_learning(dataset, total_epochs, n_users, agg, alpha)
                acc_dict[agg] = acc
                loss_dict[agg] = loss
            df_acc = pd.DataFrame.from_dict(acc_dict)
            df_loss = pd.DataFrame.from_dict(loss_dict)
            df_acc.to_csv("{}/p3_gd_acc_{}_{}_{}_{}.csv".format("outputs", dataset, alpha, n_users, total_epochs))
            df_loss.to_csv("{}/p3_gd_loss_{}_{}_{}_{}.csv".format("outputs", dataset, alpha, n_users, total_epochs))
