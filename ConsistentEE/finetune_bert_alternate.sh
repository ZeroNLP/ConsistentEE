export GLUE_DIR=/home/yhhong/Exit_itself/glue_data
#GLUE_DIR=/root/autodl-tmp/glue_data
#export CUDA_VISIBLE_DEVICES=1

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

if [ "$3" = "1" ]; then
  export MODEL_PATH="bert-base-uncased"
  export EPOCH=2
  export BATCH_SIZE=16
  export LR=2e-5
  export EVAL_BC=32
elif [ "$3" = "2" ]; then
  export MODEL_PATH="/root/autodl-tmp/early-exit-PPO-v3x-multilabel/output/$1"
  export EPOCH=1
  export BATCH_SIZE=32
  export LR=3e-4
  export EVAL_BC=1
elif [ "$3" = "3" ]; then
  export MODEL_PATH="/root/autodl-tmp/early-exit-PPO-v3x-multilabel/output/$1"
  export EPOCH=1
  export BATCH_SIZE=16
  export LR=2e-5
  export EVAL_BC=1
fi


python ./run_glue.py \
  --model_type bert \
  --model_name_or_path "$MODEL_PATH" \
  --task_name $1 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir "$GLUE_DIR/$1" \
  --max_seq_length 128 \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size $EVAL_BC \
  --learning_rate $LR \
  --save_steps $2 \
  --logging_steps $2 \
  --mode $3 \
  --num_train_epochs $EPOCH \
  --output_dir /root/autodl-tmp/early-exit-PPO-v3x-multilabel/output/$1 \
  --overwrite_output_dir \
  --overwrite_cache \
  --evaluate_during_training
