#!/bin/bash

#nohup bash finetune.sh > /home/zlchen/codes/hyh/log/0517_pretrain_0.log 2>&1 &
# mode 1
CUDA_VISIBLE_DEVICES=1 python finetune.py --base_model mode0_model_path   --learning_rate 3e-4 --alpha 0 --num_epochs 1 --output_dir lora_output_path --mode 1
CUDA_VISIBLE_DEVICES=1 python merge_code.py --base_model mode0_model_path --lora_weights lora_output_path  --output_dir mode1_model_path --mode 1

# mode 2
CUDA_VISIBLE_DEVICES=1 python finetune.py --base_model mode1_model_path   --learning_rate 3e-4 --alpha 0 --num_epochs 1 --output_dir lora_output_path --mode 2
CUDA_VISIBLE_DEVICES=1 python merge_code.py --base_model mode1_model_path --lora_weights lora_output_path  --output_dir mode2_model_path --mode 2

# mode 3
CUDA_VISIBLE_DEVICES=1 python finetune.py --base_model mode2_model_path   --learning_rate 3e-5 --alpha 0 --num_epochs 1 --output_dir lora_output_path --mode 3
CUDA_VISIBLE_DEVICES=1 python merge_code.py --base_model mode2_model_path --lora_weights lora_output_path  --output_dir mode3_model_path --mode 3