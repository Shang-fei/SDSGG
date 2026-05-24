#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml"
PROMPT_JSON="/workspace/ccloud/sf/SDSGG/analysis/predicate_factor_descriptions.json"
BASE_OUTPUT_DIR="/workspace/ccloud/sf/SDSGG/output/zero_shot_core_prompt"
LOG_DIR="/workspace/ccloud/sf/SDSGG/output/zero_shot_core_prompt_logs"

mkdir -p "${BASE_OUTPUT_DIR}" "${LOG_DIR}"

run_eval() {
  local text_source="$1"
  local exp_name="zero_shot_text_${text_source}_visual_clip_union"
  local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}.txt"

  mkdir -p "${output_dir}"
  echo "==== Evaluating ${exp_name} ===="
  echo "Output: ${output_dir}"
  echo "Log: ${log_file}"

  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 10025 \
    --nproc_per_node=1 \
    tools/relation_train_net.py \
    --eval-only \
    --config-file "${CONFIG_FILE}" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CorePromptClipPredictor \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.ENABLED False \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.PROMPT_JSON "${PROMPT_JSON}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.CORE_VISUAL_SOURCE clip_union \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.RELATION_TEXT_SOURCE "${text_source}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.ORIGINAL_CLIP_EVAL_WEIGHT 0.0 \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.DEBUG_INTERVAL 0 \
    TEST.IMS_PER_BATCH 2 \
    DTYPE "float16" \
    GLOVE_DIR ../glove \
    OUTPUT_DIR "${output_dir}" \
    > "${log_file}" 2>&1
}

run_eval decomposed
run_eval relation_of
