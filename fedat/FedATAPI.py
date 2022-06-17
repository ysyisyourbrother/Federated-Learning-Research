from .FedATSyncAggregator import SyncAggregator
from .FedATAsyncAggregator import AsyncAggregator
from .FedATTrainer import FedATTrainer
from .FedATClientManager import FedATClientManager
from .FedATServerManager import FedATServerManager
from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .FedATGroup import LatencyGroup


def FedAT_distributed(
    process_id,
    device,
    comm,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    client_data_distri_dict,
    args,
    client_config,
    model_trainer=None,
):
    if process_id == 0:
        init_server(
            args,
            device,
            comm,
            process_id,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            client_data_distri_dict,
            model_trainer,
            client_config
        )
    else:
        init_client(
            args,
            device,
            comm,
            process_id,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )


def init_server(args, device, comm, rank, model,
    train_data_num, train_data_global, test_data_global, train_data_local_dict,
    test_data_local_dict, train_data_local_num_dict, client_data_distri_dict, 
    model_trainer, client_config
):
    if model_trainer is None:
        # default model trainer is for classification problem
        model_trainer = MyModelTrainerCLS(model)
    # Set trainer id. server id is -1
    model_trainer.set_id(-1)

    # async aggregator
    async_aggregator = AsyncAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        args.client_worker_number,
        device,
        args,
        model_trainer,
    )
    latency_group = LatencyGroup(client_config, client_data_distri_dict)

    # sync aggregator 
    sync_aggregator = SyncAggregator(args, latency_group.get_group_num())

    # start the distributed training
    backend = args.backend
    server_manager = FedATServerManager(args, async_aggregator, sync_aggregator, latency_group,
                                        comm, rank, args.client_worker_number + 1, backend)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(
    args,
    device,
    comm,
    process_id,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer=None,
):
    client_index = process_id - 1
    if model_trainer is None:
        # default model trainer is for classification problem
        model_trainer = MyModelTrainerCLS(model)

    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedATTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = FedATClientManager(args, trainer, comm, process_id, 
                                        args.client_worker_number + 1, backend)
    client_manager.run()
