import logging

from common.utils import transform_list_to_tensor

class SyncAggregator():
    def __init__(self, args, group_num):
        self.args = args
        self.group_num = group_num

        self.receive_model_dict = {}
        return
    
    def clear_receive_model_dict(self, group_id):
        del self.receive_model_dict[group_id]
    
    def sync_receive_model(self, group_id, client_id, model_params, sample_num):
        logging.info(f"synchronous aggregator receive model from client{client_id}")
        if not self.receive_model_dict.get(group_id):
            self.receive_model_dict[group_id] = []
        self.receive_model_dict[group_id].append((client_id, model_params, sample_num))

        return len(self.receive_model_dict[group_id])
    
    def aggregate(self, group_id):
        receive_model = self.receive_model_dict[group_id]
        total_sample_num = 0
        model_list = []

        for i in range(len(receive_model)):
            client_id, model_params, sample_num = receive_model[i]
            if self.args.is_mobile == 1:
                model_params = transform_list_to_tensor(model_params)
            model_list.append((model_params, sample_num))
            total_sample_num += sample_num

        averaged_params, sample_num = model_list[0]
        for k in averaged_params.keys():
            for i in range(len(model_list)):
                model_params, sample_num = model_list[i]
                w = sample_num / total_sample_num
                if i == 0:
                    averaged_params[k] = model_params[k] * w
                else:
                    averaged_params[k] += model_params[k] * w

        logging.info(f"finish sync aggregate model of group-{group_id}")
        self.clear_receive_model_dict(group_id)
        return averaged_params
