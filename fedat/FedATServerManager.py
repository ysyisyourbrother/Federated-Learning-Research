import logging
import threading
import numpy as np
import time

from MLCore.communication.message import Message
from MLCore.server.server_manager import ServerManager

from .message_define import MyMessage
from common.utils import transform_list_to_tensor, transform_tensor_to_list


class FedATServerManager(ServerManager):
    def __init__(self, args, async_aggregator, sync_aggregator, latency_group, 
                 comm=None, rank=0, size=0, backend="GRPC"):
        """
        Args:
          - rank: fl_worker_index. (server=0)
          - size: total number of workers, included FL server
        """
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.async_aggregator = async_aggregator
        self.sync_aggregator = sync_aggregator
        self.latency_group = latency_group
        self.round_num = args.comm_round
        self.round_idx = 0
        self.agg_counter = [0] * self.latency_group.get_group_num()
        self.lock = threading.Lock()

    def run(self):
        super().run()


    def finish(self):
        logging.info("__finish server")
        logging.info("group aggregation counter: ", self.agg_counter)
        self.com_manager.stop_receive_message()
    

    def sample_client(self, group_id, round_idx):
        client_list = self.latency_group.get_clients(group_id)
        np.random.seed(round_idx)
        sample_client_list = np.random.choice(client_list, self.latency_group.
                                              get_group_worker_num(group_id), replace=False)  # replace: unique client index.
        logging.info(f"sample group = {group_id} client_list = {sample_client_list}", )
        return sample_client_list
    

    def send_init_msg(self):
        """ send initialize message to workers

        Send init message to all worker except server worker \
        each worker is response for one of the sampled clients.
        """
        global_model_params = self.async_aggregator.get_global_model_params()

        if self.args.is_mobile == 1:
            global_model_params = transform_tensor_to_list(global_model_params)

        # send msg to all group
        round_idx = self.round_idx
        self.async_aggregator.save_model_checkpoint(self.args.checkpoint_path)
        logging.info(f"################# Current global round is round-{round_idx} ################")
        for group_id in range(self.latency_group.get_group_num()):
            # sample client
            sample_client_list = self.sample_client(group_id, round_idx)
            self.send_message_global_model_to_group(group_id, global_model_params, sample_client_list, round_idx)
            logging.info(f"send initialize messgae to group-{group_id}")


    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)


    def handle_message_receive_model_from_client(self, msg_params):
        """ Receive and updated model.

        Use locks to ensure that the updated model is processed sequentially.
        """
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_id = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX))
        group_id =  int(msg_params.get(MyMessage.MSG_ARG_KEY_GROUP_INDEX))
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_model_round = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_MODEL_ROUND))
        local_sample_number = int(msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES))
        logging.info(f"receive message from worker-{sender_id}  client-{client_id}")

        self.lock.acquire()

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        # sync aggregate
        cur_receive_num = self.sync_aggregator.sync_receive_model(group_id, client_id, model_params, local_sample_number)
        if cur_receive_num == self.latency_group.get_group_worker_num(group_id):
            logging.info(f"receive all model of group-{group_id}")
            sync_finish_flag = True
        else:
            logging.info(f"waiting for model of group-{group_id}")
            sync_finish_flag = False
        if sync_finish_flag:
            # group aggragate counter
            self.agg_counter[group_id] += 1

            # aggregate
            new_group_model_params = self.sync_aggregator.aggregate(group_id)
            new_global_model_params = self.async_aggregator.aggregate(new_group_model_params,
                                                                      self.agg_counter[len(self.agg_counter) - 1 - group_id]/sum(self.agg_counter)) # FedAT

            if self.round_idx % self.args.frequency_of_checkpoint == 0:
                self.async_aggregator.save_model_checkpoint(self.args.checkpoint_path)
            
            # start the next round
            self.round_idx += 1
            round_idx = self.round_idx
            logging.info(f"################# Current global round is round-{round_idx} ################")
            if self.round_idx == self.round_num:
                self.finish()
                return

            # sample client
            sample_client_list = self.sample_client(group_id, round_idx)

            ## send new model to target group
            if self.args.is_mobile == 1:
                new_global_model_params = transform_tensor_to_list(new_global_model_params)
            logging.info(f"send new global model to group-{group_id}")
            
            self.lock.release()
            self.send_message_global_model_to_group(group_id, new_global_model_params, sample_client_list, round_idx)
        else:
            self.lock.release()
            
    
    def send_message_global_model_to_group(self, group_idx, new_global_model_params, sample_client_list, round_idx):
        group_worker_indexes = self.latency_group.get_group_workers(group_idx)
        assert len(sample_client_list) == len(group_worker_indexes)
        for worker_id, client_id in zip(group_worker_indexes, sample_client_list):
            delay_time = self.latency_group.get_client_delay(client_id)
            logging.info(f"send global model to worker-{worker_id}  client-{client_id}.")
            self.send_message_global_model_to_client(worker_id, new_global_model_params, 
                                                     client_id, group_idx, round_idx, delay_time)


    def send_message_global_model_to_client(self, receive_id, global_model_params, client_index, group_idx, round_idx, delay_time=0):
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        # Why str? np.int64 is not JSON serializable
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_GROUP_INDEX, group_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_MODEL_ROUND, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_DELAY_TIME, str(delay_time))
        self.send_message(message)