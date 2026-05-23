#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml"
PROMPT_JSON="/workspace/ccloud/sf/SDSGG/analysis/predicate_factor_descriptions.json"
BASE_OUTPUT_DIR="/workspace/ccloud/sf/SDSGG/output/core_prompt_ablation"
LOG_DIR="/workspace/ccloud/sf/SDSGG/output/core_prompt_ablation_logs"

mkdir -p "${BASE_OUTPUT_DIR}" "${LOG_DIR}"

run_exp() {
  local text_source="$1"
  local visual_source="$2"
  local exp_name="text_${text_source}_visual_${visual_source}"
  local output_dir="${BASE_OUTPUT_DIR}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}.txt"

  mkdir -p "${output_dir}"
  echo "==== Running ${exp_name} ===="
  echo "Output: ${output_dir}"
  echo "Log: ${log_file}"

  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 10024 \
    --nproc_per_node=1 \
    tools/relation_train_net.py \
    --config-file "${CONFIG_FILE}" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR CorePromptClipPredictor \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.ENABLED False \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.PROMPT_JSON "${PROMPT_JSON}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.CORE_VISUAL_SOURCE "${visual_source}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.RELATION_TEXT_SOURCE "${text_source}" \
    MODEL.ROI_RELATION_HEAD.LOW_RANK_TEXT.DEBUG_INTERVAL 100 \
    SOLVER.IMS_PER_BATCH 4 \
    TEST.IMS_PER_BATCH 2 \
    DTYPE "float16" \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 4000 \
    SOLVER.CHECKPOINT_PERIOD 4000 \
    GLOVE_DIR ../glove \
    OUTPUT_DIR "${output_dir}" \
    > "${log_file}" 2>&1
}

run_exp decomposed mva
run_exp decomposed clip_union
run_exp relation_of mva
run_exp relation_of clip_union
