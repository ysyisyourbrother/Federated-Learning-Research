import copy
import logging
import random
import torch

class FedAsyncAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        """
        Args:
          - worker_num: number of client worker. (exclude server workers)
        """
        self.trainer = model_trainer

        self.args = args
        self.alpha = args.aggregate_alpha
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)
    
    def save_model_checkpoint(self, path):
        self.trainer.save_model_checkpoint(path)

    def async_aggregate(self, model_params, cur_round, client_model_round):
        """Asynchronously aggregate updated model received from clients.

        Args:
          - model_params: updated model parameters
          - cur_round: current round
          - client_model_round: Original version(round) of model before updated by client
        """
        def staleness(t, tao):
            """staleness function use in fedasync"""
            # TODO: return 1
            # return 1        # constant staleness
            return 1/(t-tao+1)

        alpha_t = self.alpha * staleness(cur_round, client_model_round)
        cur_global_model_params = self.get_global_model_params()
        for k in model_params.keys():
            model_params[k] = cur_global_model_params[k] * (1-alpha_t) + model_params[k] * alpha_t
        
        self.set_global_model_params(model_params)
        return model_params

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def test_on_server_for_all_clients(self, round_idx):
        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_data_local_dict, self.device, self.args):
            return

        if round_idx % self.args.frequency_of_checkpoint == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################test_on_server_for_all_clients : round{}".format(round_idx))
            # test on training data
            # train_num_samples = []
            # train_tot_corrects = []
            # train_losses = []
            # for client_idx in range(self.args.client_num_in_total):
            #     metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
            #     train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
            #     train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            #     train_num_samples.append(copy.deepcopy(train_num_sample))
            #     train_losses.append(copy.deepcopy(train_loss))

            #     # TODO: only test the first client
            #     # break

            # train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            # train_loss = sum(train_losses) / sum(train_num_samples)
            # stats = {'training_acc': train_acc, 'training_loss': train_loss}
            # logging.info(stats)

            # test on testing/validation data
            test_num_samples = []
            test_tot_corrects = []
            test_losses = []

            if round_idx == self.args.comm_round - 1:
                metrics = self.trainer.test(self.test_global, self.device, self.args)
            else:
                metrics = self.trainer.test(self.val_global, self.device, self.args)
                
            test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
                'test_loss']
            test_tot_corrects.append(copy.deepcopy(test_tot_correct))
            test_num_samples.append(copy.deepcopy(test_num_sample))
            test_losses.append(copy.deepcopy(test_loss))

            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(stats)
