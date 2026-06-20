#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml}"
CHECKPOINT="${CHECKPOINT:-output/primitive_stage1_base/primitive_stage1/model.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/primitive_stage1_base/primitive_stage1/ablation}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

PROMPT_MODES=("triplet_only" "primitive_only" "full")
EVAL_SPLITS=(
  "val:base"
  "test:novel"
)

COMMON_ARGS=(
  --config-file "${CONFIG_FILE}"
  --checkpoint "${CHECKPOINT}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
)

if [[ -n "${DEVICE}" ]]; then
  COMMON_ARGS+=(--device "${DEVICE}")
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  COMMON_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

mkdir -p "${OUTPUT_ROOT}"

for split_part in "${EVAL_SPLITS[@]}"; do
  split="${split_part%%:*}"
  predicate_part="${split_part##*:}"

  for prompt_mode in "${PROMPT_MODES[@]}"; do
    output_dir="${OUTPUT_ROOT}/eval_${split}_${predicate_part}_${prompt_mode}"
    echo "Running split=${split} predicate_part=${predicate_part} prompt_mode=${prompt_mode}"
    python3 tools/eval_primitive_stage1_vae.py \
      "${COMMON_ARGS[@]}" \
      --split "${split}" \
      --predicate-part "${predicate_part}" \
      --prompt-mode "${prompt_mode}" \
      --output-dir "${output_dir}"
  done
done

python3 - <<'PY'
import json
import os

output_root = os.environ.get(
    "OUTPUT_ROOT",
    "output/primitive_stage1_base/primitive_stage1/ablation",
)
rows = []
for name in sorted(os.listdir(output_root)):
    path = os.path.join(output_root, name, "eval_summary.json")
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        summary = json.load(f)
    retrieval = summary["retrieval"]
    rows.append(
        {
            "name": name,
            "prompt_mode": summary.get("prompt_mode", ""),
            "predicate_part": summary.get("predicate_part", ""),
            "mse": summary["mse"],
            "cosine": summary["cosine"],
            "r@1": retrieval["r@1"],
            "r@5": retrieval["r@5"],
            "r@10": retrieval["r@10"],
            "mean_rank": retrieval["mean_rank"],
            "num_samples": summary["num_samples"],
        }
    )

print("\nAblation summary")
print("name,prompt_mode,predicate_part,num_samples,mse,cosine,r@1,r@5,r@10,mean_rank")
for row in rows:
    print(
        "{name},{prompt_mode},{predicate_part},{num_samples},{mse:.6f},{cosine:.6f},{r@1:.6f},{r@5:.6f},{r@10:.6f},{mean_rank:.2f}".format(
            **row
        )
    )
PY
