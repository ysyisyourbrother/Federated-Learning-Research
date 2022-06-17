import numpy as np
import math

class LatencyGroup():
    def __init__(self, client_config, client_data_distri_dict):
        self.class_num = 10

        # config map
        self.client_num = client_config["client_num"]
        self.group_num = client_config["group_num"]
        self.group2workers = client_config["group2workers"]
        self.delay_distribute_param = client_config["delay_distribute_param"]

        self.group2clients, self.client2group = {}, {}
        self.client2latency = {}
        self.client_data_distri_dict = client_data_distri_dict

        # init latemcy_group
        self.init_group()
    

    def cal_clients_data_distribution(self, clients):
        res = np.zeros(self.class_num)
        for c in clients:
            res += self.client_data_distri_dict[c]
        return res
    

    def cal_clients_average_response_latency(self, clients):
        sum = 0
        for c in clients:
            sum += self.client2latency[c]
        return sum / len(clients)


    def init_group(self, avg_split=True, percent_lst=None): 
        """ randomly group client into different response latency group """
        client_list = list(range(self.client_num))
        np.random.shuffle(client_list)
        if avg_split:
            split_client_lst = self.num_avg_split(client_list, self.group_num)
        else:
            assert self.group_num == len(percent_lst)
            split_client_lst = self.num_unavg_split(client_list, percent_lst)
        for gid, clst in enumerate(split_client_lst):
            for c in clst:
                self.client2group[c] = gid        
                self.client2latency[c] = float(self._sample_delay(c))
            self.group2clients[gid] = clst

        # print client spliting result
        # for g in range(self.group_num):
        #     print(self.cal_clients_average_response_latency(self.group2clients[g]))
        #     print(self.cal_clients_data_distribution(self.group2clients[g]))

    
    def get_group_num(self):
        return self.group_num
    
    def get_group_workers(self, group_id):
        return self.group2workers[group_id]

    def get_group_worker_num(self, group_id):
        return len(self.group2workers[group_id])

    def get_clients(self, group_id):
        ret = self.group2clients[group_id]
        return ret
    
    def get_group(self, client_idx):
        ret = self.client2group[client_idx]
        return ret
    
    def get_client_delay(self, client_id):
        ret = self.client2latency[client_id]
        return ret
    

    def num_avg_split(self, client_list, num_split):
        print("client avg splitting.......")
        delta, r = len(client_list) // num_split, len(client_list) % num_split
        split_client_list = []
        idx, group_idx = 0, 0
        while idx < len(client_list):
            if group_idx < r:
                split_client_list.append(client_list[idx:idx+delta+1])
                idx += delta + 1
                group_idx += 1
            else:
                split_client_list.append(client_list[idx:idx+delta])
                idx += delta
                group_idx += 1
            print(f"number of noniid clients in group{group_idx-1}: {len(split_client_list[group_idx-1])}")
        return split_client_list


    def num_unavg_split(self, client_list, percent_lst):
        print("client unavg splitting.......")
        n, sum = len(client_list), 0
        client_num_lst = []
        for p in percent_lst:
            client_num_lst.append(math.floor(n * p))
            sum += math.floor(n * p) 
        client_num_lst[-1] += n - sum

        split_client_list, cur = [], 0
        for num in client_num_lst:
            split_client_list.append(client_list[cur:cur+num])
            cur += num
        return split_client_list
    

    def _sample_delay(self, client_id):
        group_id = self.get_group(client_id)
        param = self.delay_distribute_param[group_id]
        # delay = np.random.normal(param[0], param[1], 1)[0]
        delay = np.random.uniform(param[0], param[1], 1)[0]
        if delay < 0:
            delay = 0
        return delay