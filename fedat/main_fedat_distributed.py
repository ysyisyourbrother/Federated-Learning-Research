import argparse
import logging
import os
import random
import sys
import math

import numpy as np
import torch

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))

from fedat.FedATAPI import FedAT_distributed
from common.loader import load_data, create_model, load_from_yaml

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='mobilenet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default=None,
                        help='data directory')

    parser.add_argument('--user_data_class', type=int, default=10, metavar='PA',
                        help='user data class (for mnist it means classes each client own.)')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.0001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=1,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_checkpoint', type=int, default=5,
                        help='the test frequency of the algorithms')

    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint/fedat/",
                        help='Path to store checkpoints')

    parser.add_argument('--client_config_path', type=str, default="./experiment/fedat/client_config.yaml",
                        help='Path to group config file')

    parser.add_argument('--grpc_ipconfig_path', type=str, default="grpc_ipconfig.csv",
                        help='config table containing ipv4 address of grpc server')

    parser.add_argument('--fl_worker_index', type=int, default=0,
                        help='for the server, this index is 0; for other clients, this index starts from 1')

    parser.add_argument('--mu', type=float, default=0.5,
                        help='mu parameter for FedProx algorithm (default: 0.5)')

    args = parser.parse_args()
    return args


def read_client_config(config_path):
    """split clients into different group
    """
    group_config = load_from_yaml(config_path)
    try:
        client_num_in_total = group_config["client_num"]
        client_worker_number = group_config["client_worker_number"]
    except Exception as err:
        print("Client configuration missing  keyword `client_num` or `client_worker_number`")
        raise
    return client_num_in_total, client_worker_number, group_config


def cal_client_data_distri_dict(train_data_local_dict, class_num):
    client_data_distri_dict = {}
    for c, batch_data in train_data_local_dict.items():
        for bd in batch_data:
            for label in bd[1]:
                if client_data_distri_dict.get(c) is None:
                    client_data_distri_dict[c] = np.zeros(class_num)
                client_data_distri_dict[c][label] += 1
    return client_data_distri_dict


if __name__ == "__main__":
    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.client_num_in_total, args.client_worker_number, client_config = read_client_config(args.client_config_path)
    logging.info(args)

    # create checkpoint save dir
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
    # calculate the data distribution of each client 
    client_data_distri_dict = cal_client_data_distri_dict(train_data_local_dict, class_num)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=class_num)

    FedAT_distributed(args.fl_worker_index, device, None,
                            model, train_data_num, train_data_global, test_data_global,
                            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, 
                            client_data_distri_dict, args, client_config)

