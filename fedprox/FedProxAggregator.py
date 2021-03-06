import copy
import logging
import numpy as np

from common.utils import transform_list_to_tensor


class FedProxAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                 client_worker_num, device, args, model_trainer):
        self.trainer = model_trainer

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.client_worker_num = client_worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def save_model_checkpoint(self, path):
        self.trainer.save_model_checkpoint(path)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. client index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        model_list = []
        training_num = 0

        for idx in range(self.client_worker_num):
            self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        """Sample client to join each communication round. 

        Client index start from 0. Clients' fl_worker_id start from 1.
        And here we use client index to sample.

        """
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), 
                                              num_clients, replace=False)  # replace: unique client index.
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_server_for_all_clients(self, round_idx):
        if round_idx % self.args.frequency_of_checkpoint == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : round{}".format(round_idx))
            # test on training data
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []
            for client_idx in range(self.args.client_num_in_total):
                metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
                train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                # TODO: only test the first client
                # break

            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(stats)

            # test on testing/validation data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            metrics = self.trainer.test(self.test_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
