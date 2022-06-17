#!/bin/bash
set -x

FL_WORKER_INDEX=$1

GRPC_IP_CONFIG_FILE="./experiment/fedat/grpc_ipconfig.csv"
CHECKPOINT_PATH="./checkpoint/fedat"
CLIENT_CONFIG_PATH="./experiment/fedat/client_config.yaml"

# dataset
DATASET=fashion-mnist       # cifar10 or MNIST or fashion-mnist
DATA_DIR="./data"

# data balance
USER_DATA_CLASS=2     # data label keep by any one client 

# model and training related
MODEL=cnn
ROUND=200
EPOCH=3                 # Local training epoch
BATCH_SIZE=10           # Local training batch size
LR=0.001           
CLIENT_OPTIMIZER=adam
CHECKPOINT_FREQUENCY=5
MU=0.05                 # MU=0 means fedavg   

python3 ./fedat/main_fedat_distributed.py \
  --model $MODEL \
  --dataset $DATASET \
  --data_dir $DATA_DIR \
  --user_data_class $USER_DATA_CLASS \
  --frequency_of_checkpoint $CHECKPOINT_FREQUENCY \
  --checkpoint_path $CHECKPOINT_PATH \
  --client_config_path $CLIENT_CONFIG_PATH \
  --comm_round $ROUND \
  --epochs $EPOCH \
  --client_optimizer $CLIENT_OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --backend "GRPC" \
  --grpc_ipconfig_path $GRPC_IP_CONFIG_FILE \
  --fl_worker_index $FL_WORKER_INDEX \
  --mu $MU