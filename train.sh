##!/bin/bash
#nohup torchrun --nproc_per_node=2 main.py --mode 'pretrain' --model_config 'large'> pretrain_ecg_code_v4.log 2>&1 &
nohup python -u main.py --model_config 'medium' --finetune_label_ratio 0.10 --task 'within' --r 16 --seed 18 --device "cuda:3" > training_ecg_seed18_extra.log 2>&1 &
