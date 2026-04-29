#!/bin/bash
DATASET_NAME="TAG-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python finetune.py \
--name TAG-PR-RSAL \
--img_aug \
--txt_aug \
--batch_size 128 \
--dataset_name $DATASET_NAME \
--loss_names 'nitc+ritc+id' \
--num_epoch 30 \
--root_dir ../../data \
--finetune The_pretrained_checkpoint
