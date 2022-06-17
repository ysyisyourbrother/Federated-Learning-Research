import logging
import time

from MLCore.client.client_manager import ClientManager
from MLCore.communication.message import Message

from .message_define import MyMessage
from common.utils import transform_list_to_tensor, transform_tensor_to_list


class FedATClientManager(ClientManager):
    def __init__(self, args, trainer, comm=None, rank=0, size=0, backend="GRPC"):
        super().__init__(args, comm, rank, size, backend)
        self.trainer = trainer
        self.client_model_round = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def start_training(self):
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        start = int(time.time())
        logging.info("handle_message_receive_model_from_server.")
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        client_index = int(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX))
        group_id = int(msg_params.get(MyMessage.MSG_ARG_KEY_GROUP_INDEX))
        self.client_model_round = int(msg_params.get(MyMessage.MSG_ARG_KEY_GLOBAL_MODEL_ROUND))
        delay_time = float(msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_DELAY_TIME))

        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.update_model(model_params)
        self.trainer.update_dataset(client_index)
        weights, local_sample_num = self.__train()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)

        # delay time
        time.sleep(delay_time)
        print(f"########### client {client_index} delay {delay_time}s")

        logging.info(f"handle msg time: {int(time.time())-start}")
        self.send_model_to_server(0, weights, local_sample_num, 
                                  self.client_model_round, client_index, group_id)
        logging.info("send new model back to server")

        # TODO: when to quit?
        # if self.round_idx + 1 == self.round_num:
        #     self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num, 
                             client_model_round, client_index, group_id):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_MODEL_ROUND, client_model_round)
        message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_INDEX, client_index)
        message.add_params(MyMessage.MSG_ARG_KEY_GROUP_INDEX, group_id)
        self.send_message(message)

    def __train(self):
        logging.info("####### training ########### client model round: %d" % self.client_model_round)
        weights, local_sample_num = self.trainer.train()
        return weights, local_sample_num

    def __test_local(self):
        train_tot_correct, train_loss, train_num_sample, test_tot_correct, \
        test_loss, test_num_sample = self.trainer.test()

        tot_correct = train_tot_correct + test_tot_correct
        tot_num_sample = train_num_sample + test_num_sample 
        test_acc = tot_correct / tot_num_sample

        logging.info("####### local testing ########### client model round: %d   local acc: %f" % (self.client_model_round, test_acc))
        return tot_correct, tot_num_sample
