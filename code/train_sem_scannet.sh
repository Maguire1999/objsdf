#!/bin/bash

GPU=$1
scene=$2
sparse_inv=$3

configs=confs/replica/office_${scene}_version2.conf
#confs/replica/office_3_version2.conf
NOHUP_FILE=../log/objsdf/office_${scene}_sparse_inv${sparse_inv}-GPU_${GPU}.out

export CUDA_VISIBLE_DEVICES=$1
nohup python3 ./training/exp_runner.py --conf confs/replica/office_3_version2.conf --train_type objsdf --gpu ${GPU} --sparse_inv ${sparse_inv} --timestamp 2023_05_02_18_31_00 --is_continue >$NOHUP_FILE 2>&1 &
