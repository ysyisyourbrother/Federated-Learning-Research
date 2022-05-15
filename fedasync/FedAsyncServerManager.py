import logging
import threading
import numpy as np

from .message_define import MyMessage
from common.utils import transform_list_to_tensor, transform_tensor_to_list

from MLCore.communication.message import Message
from MLCore.server.server_manager import ServerManager

class FedAsyncServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="GRPC"):
        """
        Args:
          - rank: fl_worker_index  server=0
          - size: total worker numbers (include server worker)
        """
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.lock = threading.Lock()
        self.round_num = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def client_sampling(self, num, client_num_in_total):
        """Sample client to join each communication round. 

        Client index start from 0. Clients' fl_worker_id start from 1.
        And here we use client index to sample.

        """
        sample_client_list = np.random.choice(range(client_num_in_total), num, replace=False)  # replace: unique client index.
        logging.info(f"sample client_list = {sample_client_list}")
        return sample_client_list

    def send_init_msg(self):
        # sampling clients
        global_model_params = self.aggregator.get_global_model_params()

        # Send init message to all worker except server worker \
        # each worker is response for one of the sampled clients.
        global_model_params = transform_tensor_to_list(global_model_params)
        self.aggregator.save_model_checkpoint(self.args.checkpoint_path)
        logging.info(f"############# Current global round is round-{self.round_idx}")
        round_idx = self.round_idx
        # sample clients
        sample_client_list = self.client_sampling(self.args.client_worker_number, self.args.client_num_in_total)
        assert len(sample_client_list) == self.size-1
        for process_id in range(1, self.size):
            self.send_message_global_model_to_client(process_id, global_model_params,sample_client_list[process_id - 1], round_idx)

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        """ Receive and updated model.

        Use locks to ensure that the updated model is processed sequentially.
        """
        self.lock.acquire()

        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        client_id = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX))
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_model_round = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_MODEL_ROUND))
        local_sample_number = int(msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES))
        logging.info(f"receive updated model from worker {sender_id}, client {client_id}")

        # update global model
        model_params = transform_list_to_tensor(model_params)
        new_global_model_params = self.aggregator.async_aggregate(model_params, self.round_idx, client_model_round)

        if self.round_idx % self.args.frequency_of_checkpoint == 0:
            # self.aggregator.test_on_server_for_all_clients(self.round_idx)
            self.aggregator.save_model_checkpoint(self.args.checkpoint_path)

        # start the next round
        self.round_idx += 1
        logging.info(f"############# Current global round is round-{self.round_idx}")
        if self.round_idx == self.round_num:
            # self.aggregator.test_on_server_for_all_clients(self.round_idx)
            self.finish()
            return

        ## send new model to targat client
        new_global_model_params = transform_tensor_to_list(new_global_model_params)
        # sample clients
        sample_client_list = self.client_sampling(1, self.args.client_num_in_total)
        assert len(sample_client_list) == 1
        self.send_message_global_model_to_client(sender_id, new_global_model_params, sample_client_list[0], self.round_idx)

        self.lock.release()

    def send_message_global_model_to_client(self, receive_id, global_model_params, client_index, round_idx):
        logging.info("send_message_global_model_to_client. receive_id = %d" % receive_id)
        # sample delay time
        delay_time = 0

        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        # Sample_client return np.int64, which is not JSON serializable
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_GLOBAL_MODEL_ROUND, round_idx)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_DELAY_TIME, str(delay_time))
        self.send_message(message)
