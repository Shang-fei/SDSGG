#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/visor_prism_mainline}"
DOC_FILE="${DOC_FILE:-docs/visor_prism_plan.md}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TOP_M="${TOP_M:-4}"
TEXT_ENCODE_BATCH_SIZE="${TEXT_ENCODE_BATCH_SIZE:-256}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
DEVICE="${DEVICE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

RUN_DIR="${OUTPUT_ROOT}/visor_prism"
CHECKPOINT="${RUN_DIR}/model.pth"
REPORT="${RUN_DIR}/summary.txt"
export RUN_DIR DOC_FILE

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
  echo "VISOR-PRISM Mainline"
  echo "config_file=${CONFIG_FILE}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "epochs=${EPOCHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "num_workers=${NUM_WORKERS}"
  echo "top_m=${TOP_M}"
  echo "text_encode_batch_size=${TEXT_ENCODE_BATCH_SIZE}"
  echo "skip_train=${SKIP_TRAIN}"
  echo "checkpoint=${CHECKPOINT}"
} | tee -a "${REPORT}"

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "SKIP_TRAIN=1 but checkpoint does not exist: ${CHECKPOINT}" | tee -a "${REPORT}"
    exit 1
  fi
  echo "Skipping training and using checkpoint: ${CHECKPOINT}" | tee -a "${REPORT}"
else
  run_and_log python3 tools/train_visor_prism.py \
    --config-file "${CONFIG_FILE}" \
    --predicate-part base \
    --output-dir "${OUTPUT_ROOT}" \
    --epochs "${EPOCHS}" \
    --save-every 1 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --top-m "${TOP_M}" \
    "${COMMON_DEVICE_ARGS[@]}" \
    "${COMMON_SAMPLE_ARGS[@]}"
fi

run_and_log python3 tools/eval_visor_prism.py \
  --config-file "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --split val \
  --predicate-part base \
  --candidate-part base \
  --output-dir "${RUN_DIR}/eval_val_base" \
  --batch-size "${BATCH_SIZE}" \
  --text-encode-batch-size "${TEXT_ENCODE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "${COMMON_DEVICE_ARGS[@]}" \
  "${COMMON_SAMPLE_ARGS[@]}"

run_and_log python3 tools/eval_visor_prism.py \
  --config-file "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --predicate-part novel \
  --candidate-part novel \
  --output-dir "${RUN_DIR}/eval_test_novel" \
  --batch-size "${BATCH_SIZE}" \
  --text-encode-batch-size "${TEXT_ENCODE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "${COMMON_DEVICE_ARGS[@]}" \
  "${COMMON_SAMPLE_ARGS[@]}"

run_and_log python3 tools/eval_visor_prism.py \
  --config-file "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --predicate-part novel \
  --candidate-part total \
  --output-dir "${RUN_DIR}/eval_test_novel_total" \
  --batch-size "${BATCH_SIZE}" \
  --text-encode-batch-size "${TEXT_ENCODE_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  "${COMMON_DEVICE_ARGS[@]}" \
  "${COMMON_SAMPLE_ARGS[@]}"

python3 - <<'PY' | tee -a "${REPORT}"
import json
import os
from datetime import datetime

run_dir = os.environ["RUN_DIR"]
doc_file = os.environ["DOC_FILE"]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

base = load_json(os.path.join(run_dir, "eval_val_base", "summary.json"))
novel = load_json(os.path.join(run_dir, "eval_test_novel", "summary.json"))
novel_total = load_json(os.path.join(run_dir, "eval_test_novel_total", "summary.json"))

def rows(title, data):
    out = []
    out.append("\n### {}".format(title))
    out.append("")
    out.append("| mode | count | R@1 | R@5 | R@10 | mean rank | same-SO R@10 |")
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode, row in data["modes"].items():
        same = data["same_so_raw"].get(mode, {})
        out.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {} |".format(
                mode,
                row["count"],
                row["r@1"] or 0.0,
                row["r@5"] or 0.0,
                row["r@10"] or 0.0,
                row["mean_rank"] or 0.0,
                "" if same.get("r@10") is None else "{:.4f}".format(same["r@10"]),
            )
        )
    probe = data.get("primitive_probe", {})
    out.append("")
    out.append(
        "Primitive probe: precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
            probe.get("precision", 0.0),
            probe.get("recall", 0.0),
            probe.get("f1", 0.0),
        )
    )
    return "\n".join(out)

block = []
block.append("\n## Run {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
block.append("")
block.append("Output: `{}`".format(run_dir))
block.append(rows("Validation Base Standalone", base))
block.append(rows("Test Novel Standalone (novel candidates)", novel))
block.append(rows("Test Novel Standalone (total candidates)", novel_total))
block.append("")
block = "\n".join(block)

with open(doc_file, "a") as f:
    f.write(block)

print("\n=== VISOR-PRISM Summary ===")
print(block)
print("Updated {}".format(doc_file))
PY

echo "" | tee -a "${REPORT}"
echo "Report saved to ${REPORT}" | tee -a "${REPORT}"
