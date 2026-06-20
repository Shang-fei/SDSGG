#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/primitive_stage1_mainline}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_CTX="${N_CTX:-2}"
BIAS_MODE="${BIAS_MODE:-global}"
TEXT_MODE="${TEXT_MODE:-subject_object}"
DEVICE="${DEVICE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

RUN_DIR="${OUTPUT_ROOT}/primitive_stage1"
CHECKPOINT="${RUN_DIR}/model.pth"
REPORT="${RUN_DIR}/summary.txt"
export RUN_DIR

mkdir -p "${RUN_DIR}"
: > "${REPORT}"

run_and_log() {
  echo "" | tee -a "${REPORT}"
  echo ">>> $*" | tee -a "${REPORT}"
  "$@" 2>&1 | tee -a "${REPORT}"
}

COMMON_DEVICE_ARGS=()
if [[ -n "${DEVICE}" ]]; then
  COMMON_DEVICE_ARGS+=(--device "${DEVICE}")
fi

COMMON_SAMPLE_ARGS=()
if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_SAMPLE_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

{
  echo "Primitive Stage 1 Mainline"
  echo "config_file=${CONFIG_FILE}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "epochs=${EPOCHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "num_workers=${NUM_WORKERS}"
  echo "text_mode=${TEXT_MODE}"
  echo "n_ctx=${N_CTX}"
  echo "bias_mode=${BIAS_MODE}"
  echo "checkpoint=${CHECKPOINT}"
} | tee -a "${REPORT}"

run_and_log python3 tools/train_primitive_stage1_vae.py \
  --config-file "${CONFIG_FILE}" \
  --predicate-part base \
  --output-dir "${OUTPUT_ROOT}" \
  --epochs "${EPOCHS}" \
  --save-every 1 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --text-mode "${TEXT_MODE}" \
  --n-ctx "${N_CTX}" \
  --bias-mode "${BIAS_MODE}" \
  "${COMMON_DEVICE_ARGS[@]}" \
  "${COMMON_SAMPLE_ARGS[@]}"

run_and_log python3 tools/eval_primitive_stage1_vae.py \
  --config-file "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --split val \
  --predicate-part base \
  --prompt-mode full \
  --output-dir "${RUN_DIR}/eval_val_base" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "${COMMON_DEVICE_ARGS[@]}"

run_and_log python3 tools/eval_primitive_stage1_vae.py \
  --config-file "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --predicate-part novel \
  --prompt-mode full \
  --output-dir "${RUN_DIR}/eval_test_novel" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "${COMMON_DEVICE_ARGS[@]}"

run_and_log python3 tools/eval_primitive_stage1_generation.py \
  --config-file "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --predicate-part novel \
  --num-gen-per-triplet 32 \
  --max-triplets 200 \
  --min-real-per-triplet 2 \
  --output-dir "${RUN_DIR}/generation_test_novel" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "${COMMON_DEVICE_ARGS[@]}"

python3 - <<'PY' | tee -a "${REPORT}"
import json
import os

run_dir = os.environ.get("RUN_DIR", "output/primitive_stage1_mainline/primitive_stage1")

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

base = load_json(os.path.join(run_dir, "eval_val_base", "eval_summary.json"))
novel = load_json(os.path.join(run_dir, "eval_test_novel", "eval_summary.json"))
gen = load_json(os.path.join(run_dir, "generation_test_novel", "generation_summary.json"))

print("\n=== Mainline Summary ===")
print("Reconstruction")
print("split,part,text_mode,n_ctx,bias_mode,num_samples,mse,cosine,r@10,same_pred_r@10")
for name, row in [("val", base), ("test", novel)]:
    hard = row.get("hard_retrieval", {}).get("same_predicate", {})
    print(
        "{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},{}".format(
            name,
            row.get("predicate_part"),
            row.get("text_mode"),
            row.get("n_ctx"),
            row.get("bias_mode"),
            row.get("num_samples"),
            row.get("mse"),
            row.get("cosine"),
            row.get("retrieval", {}).get("r@10"),
            "" if hard.get("r@10") is None else "{:.6f}".format(hard.get("r@10")),
        )
    )

print("\nGeneration")
print("split,part,num_real,num_triplets,num_gen,same_triplet_mean,same_triplet_max,coverage,pairwise_cos_mean,pairwise_cos_std")
print(
    "{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}".format(
        gen.get("split"),
        gen.get("predicate_part"),
        gen.get("num_real_samples"),
        gen.get("num_eval_triplets"),
        gen.get("num_gen_per_triplet"),
        gen.get("same_triplet_mean_cos"),
        gen.get("same_triplet_max_cos"),
        gen.get("coverage_real_to_generated"),
        gen.get("generated_pairwise_cos_mean"),
        gen.get("generated_pairwise_cos_std"),
    )
)
PY

echo "" | tee -a "${REPORT}"
echo "Report saved to ${REPORT}" | tee -a "${REPORT}"
