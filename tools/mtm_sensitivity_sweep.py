#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime


METRIC_RE = re.compile(
    r"SGG eval:\s+"
    r"(?P<k1>[A-Za-z\-]+)\s*@\s*20:\s*(?P<v20>[0-9.]+);\s+"
    r"(?P<k2>[A-Za-z\-]+)\s*@\s*50:\s*(?P<v50>[0-9.]+);\s+"
    r"(?P<k3>[A-Za-z\-]+)\s*@\s*100:\s*(?P<v100>[0-9.]+);\s+"
    r"for mode=(?P<mode>[^,]+), type=(?P<type>[^.]+)"
)

DETAIL_RE = re.compile(r"\(([^:]+):([0-9.]+)\)")


def parse_bool_values(value):
    if value == "both":
        return [False, True]
    if value == "on":
        return [True]
    if value == "off":
        return [False]
    raise ValueError("expected one of: on, off, both")


def bool_str(value):
    return "on" if value else "off"


def parse_metrics(text):
    metrics = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = METRIC_RE.search(lines[i])
        if not match:
            i += 1
            continue

        metric_type = match.group("type").strip()
        prefix = {
            "Recall(Main)": "R",
            "No Graph Constraint Recall(Main)": "ng_R",
            "Zero Shot Recall": "zR",
            "No Graph Constraint Zero Shot Recall": "ng_zR",
            "Mean Recall": "mR",
            "No Graph Constraint Mean Recall": "ng_mR",
            "TopK Accuracy": "A",
        }.get(metric_type, metric_type.replace(" ", "_"))

        metrics[prefix + "@20"] = float(match.group("v20"))
        metrics[prefix + "@50"] = float(match.group("v50"))
        metrics[prefix + "@100"] = float(match.group("v100"))

        if prefix in {"mR", "ng_mR"}:
            for j in range(i + 1, min(i + 8, len(lines))):
                details = DETAIL_RE.findall(lines[j])
                if details:
                    metrics[prefix + "_details"] = {
                        name: float(value) for name, value in details
                    }
                    break
        i += 1
    return metrics


def build_test_command(args, run_output_dir, split_name, part, mtm_inference, relationness, mtm_weight):
    command = [
        sys.executable,
        args.test_script,
        "--config-file",
        args.config_file,
        "DATASETS.TO_TEST",
        split_name,
        "OV_SETTING.TEST_PART",
        part,
        "MODEL.ROI_RELATION_HEAD.MTM.USE_INFERENCE",
        str(mtm_inference),
        "MODEL.ROI_RELATION_HEAD.MTM.USE_RELATIONNESS_INFERENCE",
        str(relationness),
        "MODEL.ROI_RELATION_HEAD.MTM.INFERENCE_WEIGHT",
        str(mtm_weight),
        "OUTPUT_DIR",
        run_output_dir,
    ]
    if args.weight:
        command.extend(["MODEL.WEIGHT", args.weight])
    command.extend(args.opts)
    return command


def run_command(command, log_path, dry_run=False):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_file:
        log_file.write("$ " + " ".join(command) + "\n\n")
        log_file.flush()
        if dry_run:
            return 0, ""
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            output_lines.append(line)
        return proc.wait(), "".join(output_lines)


def write_summary_csv(rows, path):
    if not rows:
        return
    fieldnames = [
        "target",
        "split",
        "part",
        "mtm_inference",
        "relationness",
        "mtm_weight",
        "returncode",
        "log_path",
        "R@20",
        "R@50",
        "R@100",
        "ng_R@20",
        "ng_R@50",
        "ng_R@100",
        "zR@20",
        "zR@50",
        "zR@100",
        "ng_zR@20",
        "ng_zR@50",
        "ng_zR@100",
        "mR@20",
        "mR@50",
        "mR@100",
        "ng_mR@20",
        "ng_mR@50",
        "ng_mR@100",
        "A@20",
        "A@50",
        "A@100",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(
        description="Run MTM/relationness inference sensitivity sweeps."
    )
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weight", default="")
    parser.add_argument("--test-script", default="tools/relation_test_net.py")
    parser.add_argument("--output-dir", default="output/mtm_sensitivity")
    parser.add_argument("--mtm-inference", choices=["on", "off", "both"], default="both")
    parser.add_argument("--relationness", choices=["on", "off", "both"], default="both")
    parser.add_argument("--mtm-weights", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--base-split", default="val")
    parser.add_argument("--novel-split", default="test")
    parser.add_argument("--base-part", default="base")
    parser.add_argument("--novel-part", default="novel")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--skip-novel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        help="Extra config overrides appended to every relation_test_net.py call.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(sweep_dir, exist_ok=True)

    with open(os.path.join(sweep_dir, "sweep_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    mtm_values = parse_bool_values(args.mtm_inference)
    relationness_values = parse_bool_values(args.relationness)
    targets = []
    if not args.skip_base:
        targets.append(("base", args.base_split, args.base_part))
    if not args.skip_novel:
        targets.append(("novel", args.novel_split, args.novel_part))

    rows = []
    for target_name, split_name, part in targets:
        for relationness in relationness_values:
            for mtm_inference in mtm_values:
                weights = args.mtm_weights if mtm_inference else [0.0]
                for mtm_weight in weights:
                    run_name = (
                        target_name
                        + "_"
                        + split_name
                        + "_rel-"
                        + bool_str(relationness)
                        + "_mtm-"
                        + bool_str(mtm_inference)
                        + "_w-"
                        + str(mtm_weight).replace(".", "p")
                    )
                    run_output_dir = os.path.join(sweep_dir, run_name, "model_output")
                    log_path = os.path.join(sweep_dir, run_name, "eval.log")
                    command = build_test_command(
                        args,
                        run_output_dir,
                        split_name,
                        part,
                        mtm_inference,
                        relationness,
                        mtm_weight,
                    )
                    print("\n==> Running", run_name)
                    returncode, output = run_command(command, log_path, args.dry_run)
                    metrics = parse_metrics(output)
                    row = {
                        "target": target_name,
                        "split": split_name,
                        "part": part,
                        "mtm_inference": mtm_inference,
                        "relationness": relationness,
                        "mtm_weight": mtm_weight,
                        "returncode": returncode,
                        "log_path": log_path,
                    }
                    row.update(metrics)
                    rows.append(row)
                    result_path = os.path.join(sweep_dir, run_name, "metrics.json")
                    with open(result_path, "w") as f:
                        json.dump(row, f, indent=2, sort_keys=True)
                    write_summary_csv(rows, os.path.join(sweep_dir, "summary.csv"))
                    with open(os.path.join(sweep_dir, "summary.json"), "w") as f:
                        json.dump(rows, f, indent=2, sort_keys=True)
                    if returncode != 0:
                        raise RuntimeError("Run failed: {}. See {}".format(run_name, log_path))

    print("\nSweep results saved to", sweep_dir)


if __name__ == "__main__":
    main()
