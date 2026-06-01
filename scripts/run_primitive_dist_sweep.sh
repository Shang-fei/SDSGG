#!/usr/bin/env bash
set -u

GPU="${GPU:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-12030}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG="${CONFIG:-configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml}"
GLOVE_DIR="${GLOVE_DIR:-../glove}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/ccloud/sf/SDSGG/output/concept}"
LOG_DIR="${LOG_DIR:-logs/primitive_dist_sweep}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/sweep_$(date +%Y%m%d_%H%M%S).log}"

IMS_PER_BATCH="${IMS_PER_BATCH:-4}"
TEST_IMS_PER_BATCH="${TEST_IMS_PER_BATCH:-2}"
MAX_ITER="${MAX_ITER:-16000}"
VAL_PERIOD="${VAL_PERIOD:-4000}"
CHECKPOINT_PERIOD="${CHECKPOINT_PERIOD:-8000}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

NAMES=(
  "nodist_of05"
  "dist_s01_nosample_of05"
  "dist_s02_nosample_of05"
  "dist_s01_sample_of05"
)

DIST_ENABLED=(False True True True)
DIST_SAMPLE=(False False False True)
SHIFT_SCALE=(0.0 0.1 0.2 0.1)
NOISE_SCALE=(0.0 0.02 0.02 0.02)
OBJECT_FILTER_WEIGHT=(0.5 0.5 0.5 0.5)

echo "log_file=${LOG_FILE}" >> "${LOG_FILE}"
echo "config=${CONFIG}" >> "${LOG_FILE}"
echo "output_root=${OUTPUT_ROOT}" >> "${LOG_FILE}"
echo >> "${LOG_FILE}"

for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  port=$((MASTER_PORT_BASE + idx))
  output_dir="${OUTPUT_ROOT}/${name}"

  echo "===== ${name} =====" >> "${LOG_FILE}"
  echo "port=${port}" >> "${LOG_FILE}"
  echo "output_dir=${output_dir}" >> "${LOG_FILE}"
  echo "distribution_enabled=${DIST_ENABLED[$idx]} distribution_sample=${DIST_SAMPLE[$idx]} shift=${SHIFT_SCALE[$idx]} noise=${NOISE_SCALE[$idx]} object_filter=${OBJECT_FILTER_WEIGHT[$idx]}" >> "${LOG_FILE}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m torch.distributed.launch \
    --master_port "${port}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    tools/relation_train_net.py \
    --config-file "${CONFIG}" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR PrimitiveLowRankClipPredictor \
    MODEL.ROI_RELATION_HEAD.PRIMITIVE_TEXT.DISTRIBUTION_ENABLED "${DIST_ENABLED[$idx]}" \
    MODEL.ROI_RELATION_HEAD.PRIMITIVE_TEXT.DISTRIBUTION_SAMPLE "${DIST_SAMPLE[$idx]}" \
    MODEL.ROI_RELATION_HEAD.PRIMITIVE_TEXT.DISTRIBUTION_SHIFT_SCALE "${SHIFT_SCALE[$idx]}" \
    MODEL.ROI_RELATION_HEAD.PRIMITIVE_TEXT.DISTRIBUTION_NOISE_SCALE "${NOISE_SCALE[$idx]}" \
    MODEL.ROI_RELATION_HEAD.PRIMITIVE_TEXT.OBJECT_FILTER_WEIGHT "${OBJECT_FILTER_WEIGHT[$idx]}" \
    SOLVER.IMS_PER_BATCH "${IMS_PER_BATCH}" \
    TEST.IMS_PER_BATCH "${TEST_IMS_PER_BATCH}" \
    DTYPE "float16" \
    SOLVER.MAX_ITER "${MAX_ITER}" \
    SOLVER.VAL_PERIOD "${VAL_PERIOD}" \
    SOLVER.CHECKPOINT_PERIOD "${CHECKPOINT_PERIOD}" \
    GLOVE_DIR "${GLOVE_DIR}" \
    OUTPUT_DIR "${output_dir}" \
    >> "${LOG_FILE}"

  status=$?
  echo "exit_code=${status}" >> "${LOG_FILE}"
  echo >> "${LOG_FILE}"
done

echo "done" >> "${LOG_FILE}"
