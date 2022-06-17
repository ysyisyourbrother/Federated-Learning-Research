class FedATTrainer(object):

    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict,
                 train_data_num, device, args, model_trainer):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self):
        self.trainer.train(self.train_local, self.device, self.args)
        weights = self.trainer.get_model_params()
        return weights, self.local_sample_number

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
                                                          train_metrics['test_total'], train_metrics['test_loss']

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
                                                          test_metrics['test_total'], test_metrics['test_loss']
        return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample
    
    def test_client_by_group(self, group2clients):
        group_acc = {}
        for group_id, clients in group2clients.items():
            group_tot_correct, group_num_sample = 0, 0
            for client_idx in clients:
                metrics = self.trainer.test(self.test_data_local_dict[client_idx], self.device, self.args)
                group_tot_correct += metrics['test_correct']
                group_num_sample += metrics['test_total']
            group_acc[group_id] = group_tot_correct/group_num_sample
        return group_acc