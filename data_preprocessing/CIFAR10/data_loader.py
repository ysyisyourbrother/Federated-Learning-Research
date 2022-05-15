import torch
import numpy as np
import time
import os
import sys

import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
sys.path.insert(0, "./")


DATA_CLASS = 10

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            # print(i, i+delta+1)
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


def choose_digit_random(split_data_lst, user_data_class):
    np.random.seed(6)
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, user_data_class, replace=False).tolist()
    except:
        print("available_digit:", available_digit)
    return lst


def choose_digit_inturn(user, user_data_class):
    """Choose digit in turn for all user client.
    """
    idx = user % DATA_CLASS
    return [(idx+i)%DATA_CLASS for i in range(user_data_class)]


def main(root_dir, user_data_class, client_num, num_split):
    train_transform, test_transform = _data_transforms_cifar10()

    # Get Cifar10 data, normalize, and divide by lebel
    cifar_trainset = torchvision.datasets.CIFAR10(root_dir, transform = train_transform, download=True, train=True)
    cifar_testset = torchvision.datasets.CIFAR10(root_dir, transform = test_transform, download=True, train=False)
    ori_train_data, ori_train_target = cifar_trainset.data, np.array(cifar_trainset.targets)
    ori_test_data, ori_test_target = cifar_testset.data, np.array(cifar_testset.targets)

    # divide train data by label
    cifar10_traindata = []
    for number in range(DATA_CLASS):
        idx = ori_train_target == number
        cifar10_traindata.append(ori_train_data[idx])
    split_cifar10_traindata = []
    for digit in cifar10_traindata:
        split_cifar10_traindata.append(data_split(digit, num_split))

    # divide test data by label
    cifar10_testdata = []
    for number in range(DATA_CLASS):
        idx = ori_test_target == number
        cifar10_testdata.append(ori_test_data[idx])
    split_cifar10_testdata = []
    for digit in cifar10_testdata:
        split_cifar10_testdata.append(data_split(digit, num_split))


    # data_distribution = np.array([len(v) for v in cifar10_traindata])
    # data_distribution = np.round(data_distribution / data_distribution.sum(), 3)

    # Assign train samples to each user
    train_X = [[] for _ in range(client_num)]
    train_y = [[] for _ in range(client_num)]
    test_X = [[] for _ in range(client_num)]
    test_y = [[] for _ in range(client_num)]
    for user in range(client_num):
        print(user, np.array([len(v) for v in split_cifar10_traindata]))

        for d in choose_digit_inturn(user, user_data_class):
            l = len(split_cifar10_traindata[d][-1])
            train_X[user].extend(split_cifar10_traindata[d].pop())
            train_y[user].extend(d * np.ones(l))
            # train_X[user] += split_cifar10_traindata[d].pop().tolist()
            # train_y[user] += (d * np.ones(l)).tolist()

            l = len(split_cifar10_testdata[d][-1])
            test_X[user].extend(split_cifar10_testdata[d].pop())
            test_y[user].extend((d * np.ones(l)))
            # test_X[user] += split_cifar10_testdata[d].pop().tolist()
            # test_y[user] += (d * np.ones(l)).tolist()

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup users
    for i in range(client_num):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    return train_data, test_data


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i + batch_size]
        batched_y = data_y[i:i + batch_size]
        batched_x = torch.from_numpy(np.asarray(batched_x)).float()
        batched_y = torch.from_numpy(np.asarray(batched_y)).long()
        batch_data.append((batched_x, batched_y))
    return batch_data


def load_partition_data_cifar10(data_dir, client_num, user_data_class, batch_size):
    """Partition CIFAR10 data with iid or non-iid.

    Args:
      - user_data_class: number of class each user own.
      - net_dataidx_map: {"client_index": [1,5,7,21,...]}, map from client_idx to data sample index.
      - traindata_cls_counts: {client1: {"class 0": 5, "class 1": 1, ...}, ...}
      - user_data_class: 10 means balanced data
    
    Returns:
      - train_local_data_num: Number of training samples on each client
      - train_data_local_dict: Training samples on each client
    """

    assert user_data_class <= 10
    assert client_num % DATA_CLASS == 0
    num_split = int(client_num * user_data_class / DATA_CLASS)

    train_data_dict, test_data_dict = main(data_dir, user_data_class, client_num, num_split)
    train_data = train_data_dict["user_data"]
    test_data = test_data_dict["user_data"]

    train_data_num = 0
    test_data_num = 0
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = list()
    test_data_global = list()
    for c in range(client_num):
        print("processing client", c)
        user_train_data_num = len(train_data[c]['x'])
        user_test_data_num = len(test_data[c]['x'])
        train_data_num += user_train_data_num
        test_data_num += user_test_data_num
        train_data_local_num_dict[c] = user_train_data_num

        # transform to batches
        train_batch = batch_data(train_data[c], batch_size)
        test_batch = batch_data(test_data[c], batch_size)

        # index using client index
        train_data_local_dict[c] = train_batch
        test_data_local_dict[c] = test_batch
        train_data_global += train_batch
        test_data_global += test_batch

    return train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, DATA_CLASS


if __name__ == '__main__':
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, DATA_CLASS = load_partition_data_cifar10("./data/cifar10", 500, 2, 32)