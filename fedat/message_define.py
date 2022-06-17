class MyMessage(object):
    """
        message type definition
    """
    # server to client
    MSG_TYPE_S2C_INIT_CONFIG = 1
    MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT = 2

    # client to server
    MSG_TYPE_C2S_SEND_MODEL_TO_SERVER = 3
    MSG_TYPE_C2S_SEND_STATS_TO_SERVER = 4

    # profiler to server
    MSG_TYPE_P2S_SEND_MODEL_TO_SERVER = 5
    # server to profiler
    MSG_TYPE_S2P_SEND_MODEL_TO_PROFILER = 6 

    MSG_ARG_KEY_TYPE = "msg_type"
    MSG_ARG_KEY_SENDER = "sender"
    MSG_ARG_KEY_RECEIVER = "receiver"

    """
        message payload keywords definition
    """
    MSG_ARG_KEY_NUM_SAMPLES = "num_samples"
    MSG_ARG_KEY_MODEL_PARAMS = "model_params"
    MSG_ARG_KEY_CLIENT_MODEL_ROUND = "client_model_round"               # Original version of model before updated by client
    MSG_ARG_KEY_GLOBAL_MODEL_ROUND = "global_model_round"               # Current version of global model before updated by client
    MSG_ARG_KEY_CLIENT_DELAY_TIME = "client_delay_time"                 # added random delay to simulate different levels of straggler effects
    MSG_ARG_KEY_CLIENT_INDEX = "client_idx"
    MSG_ARG_KEY_GROUP_INDEX = "group_idx"

    MSG_ARG_KEY_GROUP_ACC = "group_acc"
    MSG_ARG_KEY_GROUP2CLIENTS = "group2clients"

    MSG_ARG_KEY_TRAIN_CORRECT = "train_correct"
    MSG_ARG_KEY_TRAIN_ERROR = "train_error"
    MSG_ARG_KEY_TRAIN_NUM = "train_num_sample"

    MSG_ARG_KEY_TEST_CORRECT = "test_correct"
    MSG_ARG_KEY_TEST_ERROR = "test_error"
    MSG_ARG_KEY_TEST_NUM = "test_num_sample"

