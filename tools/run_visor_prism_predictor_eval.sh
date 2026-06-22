#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml}"
SDSGG_CKPT="${SDSGG_CKPT:?Set SDSGG_CKPT to the baseline relation checkpoint path.}"
VISOR_CKPT="${VISOR_CKPT:-output/visor_prism_rankloss/visor_prism/model.pth}"
ALPHA="${ALPHA:-0.05}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CANDIDATE_PART="${CANDIDATE_PART:-total}"
OUTPUT_DIR="${OUTPUT_DIR:-output/visor_prism_predictor_eval_alpha_${ALPHA}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/test.log}"

mkdir -p "${OUTPUT_DIR}"

echo "config=${CONFIG_FILE}"
echo "sdsgg_ckpt=${SDSGG_CKPT}"
echo "visor_ckpt=${VISOR_CKPT}"
echo "alpha=${ALPHA}"
echo "temperature=${TEMPERATURE}"
echo "candidate_part=${CANDIDATE_PART}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_file=${LOG_FILE}"

python3 tools/relation_test_net.py \
  --config-file "${CONFIG_FILE}" \
  MODEL.WEIGHT "${SDSGG_CKPT}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  MODEL.ROI_RELATION_HEAD.VISOR_PRISM.ENABLED True \
  MODEL.ROI_RELATION_HEAD.VISOR_PRISM.CKPT "${VISOR_CKPT}" \
  MODEL.ROI_RELATION_HEAD.VISOR_PRISM.ALPHA "${ALPHA}" \
  MODEL.ROI_RELATION_HEAD.VISOR_PRISM.TEMPERATURE "${TEMPERATURE}" \
  MODEL.ROI_RELATION_HEAD.VISOR_PRISM.CANDIDATE_PART "${CANDIDATE_PART}" \
  2>&1 | tee "${LOG_FILE}"
