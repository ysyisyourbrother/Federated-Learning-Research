import logging
from os import sync
import threading
import time

from .message_define import MyMessage
from common.utils import transform_tensor_to_list

from MLCore.communication.message import Message
from MLCore.server.server_manager import ServerManager


class FedProxServerManager(ServerManager):
    def __init__(self, args, aggregator, comm=None, rank=0, size=0, backend="GRPC"):
        """
        Args:
          - rank: fl_worker_index  server=0
          - size: client_num_per_round + 1
        """
        super().__init__(args, comm, rank, size, backend)
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def send_init_msg(self):
        # sampling clients
        client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                         self.args.client_worker_number)
        global_model_params = self.aggregator.get_global_model_params()

        # Send init message to all worker except server worker \
        # each worker is response for one of the sample clients.
        global_model_params = transform_tensor_to_list(global_model_params)

        logging.info(f"############## Current global round is round-{self.round_idx}")
        self.aggregator.save_model_checkpoint(self.args.checkpoint_path)
        for worker_id in range(1, self.size):
            self.send_message_sync_model_to_client(worker_id, global_model_params, client_indexes[worker_id - 1])

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.handle_message_receive_model_from_client)

    def handle_message_receive_model_from_client(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        client_id = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX))
        logging.info(f"receive updated model from worker {sender_id}, client {client_id}")

        self.aggregator.add_local_trained_result(sender_id - 1, model_params, local_sample_number)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            global_model_params = self.aggregator.aggregate()

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

            # sampling clients
            client_indexes = self.aggregator.client_sampling(self.round_idx, self.args.client_num_in_total,
                                                             self.args.client_worker_number)
            
            global_model_params = transform_tensor_to_list(global_model_params)

            for worker_id in range(1, self.size):
                self.send_message_sync_model_to_client(worker_id, global_model_params, client_indexes[worker_id - 1])


    def send_message_sync_model_to_client(self, receive_id, global_model_params, client_index):
        # client delay time
        delay_time = 0

        logging.info("send_message_sync_model_to_client. receive_id = %d" % receive_id)
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, str(client_index))
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_DELAY_TIME, str(delay_time))
        self.send_message(message)
