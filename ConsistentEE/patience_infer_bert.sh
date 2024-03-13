#export GLUE_DIR=/home/yhhong/Exit_itself/glue_data
GLUE_DIR=/root/autodl-tmp/glue_data
#export GLUE_DIR=/data/yhhong/glue_data
#export CUDA_VISIBLE_DEVICES=2

# 获取GPU空闲情况
GPU_INFO=$(nvidia-smi --format=csv,noheader,nounits --query-gpu=memory.free,index)

# 解析GPU信息并选择空闲最多的GPU
BEST_GPU=-1
MAX_FREE_MEMORY=0

while IFS=',' read -r MEMORY INDEX; do
  if ((MEMORY > MAX_FREE_MEMORY)); then
    MAX_FREE_MEMORY=$MEMORY
    BEST_GPU=$INDEX
  fi
done <<< "$GPU_INFO"

# 设置CUDA_VISIBLE_DEVICES以选择最佳GPU
export CUDA_VISIBLE_DEVICES=$BEST_GPU

python ./run_glue.py \
  --model_type bert \
  --model_name_or_path /root/autodl-tmp/early-exit-PPO-v3x-multilabel/output/$1 \
  --task_name $1 \
  --do_eval \
  --do_lower_case \
  --data_dir "$GLUE_DIR/$1" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --save_steps $2 \
  --logging_steps $2 \
  --mode 3 \
  --num_train_epochs 1 \
  --output_dir /root/autodl-tmp/early-exit-PPO-v3x-multilabel/output/$1 \
  --overwrite_output_dir \
  --overwrite_cache \
  --eval_all_checkpoints \
  --patience 12 #2,3,4,5,6,7,8,9,10,11,12 #65,66,67,68,69,70
