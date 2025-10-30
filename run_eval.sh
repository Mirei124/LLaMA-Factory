#!/bin/bash

ADAPTER=

model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct
adapter_name_or_path="$ADAPTER"
save_name="$ADAPTER"/eval/generated_predictions.jsonl
dataset=

cutoff_len=8192
temperature=1e-6  # vllm/sampling_params.py _SAMPLING_EPS = 1e-5
top_p=1.0
top_k=0
max_new_tokens=4096
image_max_pixels=12544
image_min_pixels=12544

vllm_config=$'
{
  "max_lora_rank": 32,
  "gpu_memory_utilization": 0.5,
  "enforce_eager": true
}'

DISABLE_VERSION_CHECK=1 python scripts/vllm_infer.py \
  --model_name_or_path="$model_name_or_path" \
  --adapter_name_or_path="$adapter_name_or_path" \
  --dataset="$dataset" \
  --template=qwen2_vl \
  --cutoff_len="$cutoff_len" \
  --vllm_config="$vllm_config" \
  --save_name="$save_name" \
  --temperature="$temperature" \
  --top_p="$top_p" \
  --top_k="$top_k" \
  --max_new_tokens="$max_new_tokens" \
  --seed=42 \
  --pipeline_parallel_size=1 \
  --image_max_pixels="$image_max_pixels" \
  --image_min_pixels="$image_min_pixels"
