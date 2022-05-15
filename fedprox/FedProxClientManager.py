import logging
import time

from MLCore.client.client_manager import ClientManager
from MLCore.communication.message import Message

from .message_define import MyMessage
from common.utils import transform_list_to_tensor, transform_tensor_to_list


class FedProxClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="GRPC"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        start = time.time()
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX))
        delay_time = float(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_DELAY_TIME))

        model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(int(client_index))
        self.round_idx += 1
        weights, local_sample_num = self.__train()
        # transform Tensor to list
        weights = transform_tensor_to_list(weights)

        # delay time
        time.sleep(delay_time)
        print(f"########### client {client_index} delay {delay_time}s")
        print(f"############# client {client_index}   training time: {time.time()-start}")

        self.send_model_to_server(0, weights, local_sample_num, client_index)


    def send_model_to_server(self, receive_id, weights, local_sample_num, client_index):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        weights, local_sample_num = self.trainer.train(self.round_idx)
        return weights, local_sample_num
