# Federated-Learning-Research
An implementation of federated learning research baseline methods based on [FedML-core](https://github.com/FedML-AI/FedML). This is not an implementation for only stand-alone simulation, but a distributed system that can be deployed on multiple real devices (or several docker containers on a same server), which can help researchers to explore more problems that may exist on real FL systems.

## Quick Start
Here is demo to deploy our framework on a cluster with three devices. `A` (ip: 172.17.0.4) represents the server worker, `B` (ip: 172.17.0.13) and `C` (ip: 172.17.0.12) represent two client workers in FL system. All devices must have a `python3` environment with `pytorch` and `grpc` installed. You can find the env-requirement under `requirement.txt` file.

Startup scripts for all methods are under `experiment/` directory. We first need to modify the `grpc_ipconfig.csv` file as below:
```csv
receiver_id,ip
0,172.17.0.4
1,172.17.0.13
2,172.17.0.12
```
Receiver_id represents the worker_id for each worker in FL system. Usually, worker_id of server worker is `0` and client workers' id will start from `1`.

After that, we will start the worker process on each devices. All client workers need to be started up before server worker. **Remember to execute all commands below at the root direcroy of this project:**
```
python experiment/fedprox/run_fedprox_distributed.sh $worker_id
```

Then the training process will begin and the checkpoint of global model will be saved under `checkpoint/fedprox/`. You can change the experiment settings by editing the pre-set arguments in `run_fedprox_distributed.sh`.

## Reproduced FL Algorithms
|Method|Reference|
|---|---|
|FedAvg|[McMahan et al., 2017](https://arxiv.org/abs/1602.05629)|
|FedProx|[Li et al., 2020](https://arxiv.org/abs/1812.06127)|
|FedAsync|[Wang et al., 2021](https://arxiv.org/pdf/1903.03934.pdf?ref=https://githubhelp.com)|
|FedAT|[Zheng Chai et al., 2021](https://dl.acm.org/doi/10.1145/3458817.3476211)|
|Eco-FL|Shengyuan Ye et al., 2022|
| On going         |...|

## Contacts
- Shengyuan Ye, brandonye [AT] foxmail [dot] com
- Liekang Zeng, zenglk3 [AT] mail2 [dot] sysu [dot] edu [dot] cn


