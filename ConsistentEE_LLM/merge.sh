#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python merge_code.py --base_model Llama2-7b-hf --lora_weights alpaca_lora --output_dir mode0_model_path --mode 0