"""common helper functions
"""

import logging
import yaml

def load_data(args, dataset_name):
    """Load target data according to user's configuration.
    """
    logging.info("load_data. dataset_name = %s" % dataset_name)
    if dataset_name == "mnist":
        from data_preprocessing.MNIST.data_loader import load_partition_data_mnist
        data_loader = load_partition_data_mnist
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.data_dir, args.client_num_in_total, args.user_data_class, args.batch_size)
    elif dataset_name == "fashion-mnist":
        from data_preprocessing.FashionMNIST.data_loader import load_partition_data_fashion_mnist
        data_loader = load_partition_data_fashion_mnist
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.data_dir, args.client_num_in_total, args.user_data_class, args.batch_size)
    elif dataset_name == "cifar10":
        from data_preprocessing.CIFAR10.data_loader import load_partition_data_cifar10
        data_loader = load_partition_data_cifar10
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.data_dir, args.client_num_in_total, args.user_data_class, args.batch_size)
    else:
        raise ValueError(f"dataset {dataset_name} have not been supported.")

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def create_model(args, model_name, output_dim):
    """Create model according to user's configuration.
    """
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "cnn" and args.dataset == "cifar10":
        from models.cnn import CNN_OriginalFedAvg
        logging.info("CNN + Cifar10")
        model = CNN_OriginalFedAvg(True)
    elif model_name == "cnn" and args.dataset == "mnist":
        from models.cnn import CNN_DropOut
        logging.info("CNN + mnist")
        model = CNN_DropOut(True)
    elif model_name == "cnn" and args.dataset == "fashion-mnist":
        from models.cnn import CNN_DropOut
        logging.info("CNN + fashion-mnist")
        model = CNN_DropOut(True)
    else:
        raise ValueError(f"{model_name} + {args.dataset} have not been supported.")
    return model

def load_from_yaml(filepath):
    try:
        with open(filepath, mode='rb') as config_file:
            config = yaml.full_load(config_file.read())
    except IOError as e:                
        e.strerror = "Unable to load configuration file (%s)" % e.strerror
        raise
    return config