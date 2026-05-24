#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml"
PROMPT_JSON="/workspace/ccloud/sf/SDSGG/analysis/predicate_factor_descriptions.json"
BASE_OUTPUT_DIR="/workspace/ccloud/sf/SDSGG/output/core_prompt_full_ablation"
LOG_DIR="/workspace/ccloud/sf/SDSGG/output/core_prompt_full_ablation_logs"

mkdir -p "${BASE_OUTPUT_DIR}" "${LOG_DIR}"

COMMON_OPTS=(
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
  MODEL.ROI_RELATION_HEAD.PREDICTOR CorePromptClipPredictor
  MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.ENABLED False
  MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.PROMPT_JSON "${PROMPT_JSON}"
  TEST.IMS_PER_BATCH 2
  DTYPE "float16"
  GLOVE_DIR ../glove
)

run_union_zero_shot() {
  local text_source="$1"
  local exp_name="zero_shot_text_${text_source}_visual_clip_union_clip_eval_0.0"
  local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}.txt"

  mkdir -p "${output_dir}"
  echo "==== Zero-shot eval: ${exp_name} ===="
  echo "Output: ${output_dir}"
  echo "Log: ${log_file}"

  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 10025 \
    --nproc_per_node=1 \
    tools/relation_train_net.py \
    --eval-only \
    --eval-val \
    --config-file "${CONFIG_FILE}" \
    "${COMMON_OPTS[@]}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.CORE_VISUAL_SOURCE clip_union \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.RELATION_TEXT_SOURCE "${text_source}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.ORIGINAL_CLIP_EVAL_WEIGHT 0.0 \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.DEBUG_INTERVAL 0 \
    OUTPUT_DIR "${output_dir}" \
    > "${log_file}"
}

run_mva_train() {
  local text_source="$1"
  local clip_eval_weight="$2"
  local exp_name="train_text_${text_source}_visual_mva_clip_eval_${clip_eval_weight}"
  local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}.txt"

  mkdir -p "${output_dir}"
  echo "==== Train/eval: ${exp_name} ===="
  echo "Output: ${output_dir}"
  echo "Log: ${log_file}"

  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 10024 \
    --nproc_per_node=1 \
    tools/relation_train_net.py \
    --config-file "${CONFIG_FILE}" \
    "${COMMON_OPTS[@]}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.CORE_VISUAL_SOURCE mva \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.RELATION_TEXT_SOURCE "${text_source}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.ORIGINAL_CLIP_EVAL_WEIGHT "${clip_eval_weight}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.DEBUG_INTERVAL 100 \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 4000 \
    SOLVER.CHECKPOINT_PERIOD 4000 \
    OUTPUT_DIR "${output_dir}" \
    > "${log_file}"
}

TEXT_SOURCES=(decomposed relation_of)
CLIP_EVAL_WEIGHTS=(0.0 0.4 0.8)

for text_source in "${TEXT_SOURCES[@]}"; do
  run_union_zero_shot "${text_source}"
done

for text_source in "${TEXT_SOURCES[@]}"; do
  for clip_eval_weight in "${CLIP_EVAL_WEIGHTS[@]}"; do
    run_mva_train "${text_source}" "${clip_eval_weight}"
  done
done
