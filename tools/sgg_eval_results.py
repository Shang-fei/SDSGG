import re
import os
import pandas as pd


# =========================
# 每次实验只改这里
# =========================

experiment_name = "dist_v2"
output_path = "/workspace/ccloud/sf/SDSGG/sgg_eval_results.xlsx"

log_text = r"""
====================================================================================================
SGG eval:     R @ 20: 0.1246;     R @ 50: 0.1793;     R @ 100: 0.2166;  for mode=predcls, type=Recall(Main).
SGG eval:  ng-R @ 20: 0.1673;  ng-R @ 50: 0.2853;  ng-R @ 100: 0.4016;  for mode=predcls, type=No Graph Constraint Recall(Main).
SGG eval:    zR @ 20: 0.1565;    zR @ 50: 0.2539;    zR @ 100: 0.3257;  for mode=predcls, type=Zero Shot Recall.
SGG eval: ng-zR @ 20: 0.1659; ng-zR @ 50: 0.2829; ng-zR @ 100: 0.3954;  for mode=predcls, type=No Graph Constraint Zero Shot Recall.
SGG eval:    mR @ 20: 0.0593;    mR @ 50: 0.0995;    mR @ 100: 0.1277;  for mode=predcls, type=Mean Recall.
----------------------- Details ------------------------
(above:0.0000) (against:0.0000) (at:0.0000) (attached to:0.0000) (behind:0.1883) (belonging to:0.0000) (between:0.0000) (carrying:0.0000) (covered in:0.0714) (covering:0.0000) (for:0.0000) (from:0.0000) (hanging from:0.0000) (has:0.6885) (holding:0.1757) (in:0.0047) (in front of:0.0000) (looking at:0.0000) (made of:0.0000) (near:0.2244) (of:0.5733) (on:0.0490) (over:0.0061) (parked on:0.4680) (playing:0.0000) (riding:0.5402) (sitting on:0.4485) (standing on:0.0000) (to:0.0000) (under:0.0000) (walking on:0.0000) (watching:0.3627) (wearing:0.6678) (wears:0.0000) (with:0.0006) 
--------------------------------------------------------
"""


def extract_metric(log_text, metric_name):
    pattern = (
        rf"{re.escape(metric_name)}\s*@\s*20:\s*([\d.]+);\s*"
        rf"{re.escape(metric_name)}\s*@\s*50:\s*([\d.]+);\s*"
        rf"{re.escape(metric_name)}\s*@\s*100:\s*([\d.]+)"
    )

    match = re.search(pattern, log_text)

    if match is None:
        return {}

    return {
        f"{metric_name}@20": float(match.group(1)),
        f"{metric_name}@50": float(match.group(2)),
        f"{metric_name}@100": float(match.group(3)),
    }


def extract_details_as_wide_column(log_text, experiment_name):
    """
    输出格式：

    predicate      exp_name
    across         0.0952
    along          0.0000
    and            0.0000
    """

    detail_block_pattern = r"Details\s*-+\s*(.*?)\s*-+"
    block_match = re.search(detail_block_pattern, log_text, flags=re.S)

    if block_match is None:
        return pd.DataFrame(columns=["predicate", experiment_name])

    detail_text = block_match.group(1)

    pair_pattern = r"\((.*?):([\d.]+)\)"
    pairs = re.findall(pair_pattern, detail_text)

    rows = []
    for predicate, recall in pairs:
        rows.append({
            "predicate": predicate.strip(),
            experiment_name: float(recall)
        })

    return pd.DataFrame(rows)


def append_result_to_excel(log_text, experiment_name, output_path):
    # =========================
    # 1. summary metrics
    # =========================

    metrics = {
        "experiment_name": experiment_name
    }

    for metric_name in ["R", "mR"]:
        metrics.update(extract_metric(log_text, metric_name))

    new_summary_df = pd.DataFrame([metrics])

    # =========================
    # 2. predicate details，宽表格式
    # =========================

    new_details_df = extract_details_as_wide_column(
        log_text=log_text,
        experiment_name=experiment_name
    )

    # =========================
    # 3. 如果 Excel 已存在，读取旧表
    # =========================

    if os.path.exists(output_path):
        try:
            old_summary_df = pd.read_excel(
                output_path,
                sheet_name="summary_metrics"
            )
        except Exception:
            old_summary_df = pd.DataFrame()

        try:
            old_details_df = pd.read_excel(
                output_path,
                sheet_name="predicate_details"
            )
        except Exception:
            old_details_df = pd.DataFrame()

        # 追加 summary，一行一个实验
        summary_df = pd.concat(
            [old_summary_df, new_summary_df],
            ignore_index=True
        )

        # predicate details 按 predicate 合并
        if old_details_df.empty:
            details_df = new_details_df
        else:
            # 如果同名实验已经存在，先删除旧列，避免重复
            if experiment_name in old_details_df.columns:
                old_details_df = old_details_df.drop(columns=[experiment_name])

            details_df = pd.merge(
                old_details_df,
                new_details_df,
                on="predicate",
                how="outer"
            )

    # =========================
    # 4. 如果 Excel 不存在，新建
    # =========================

    else:
        summary_df = new_summary_df
        details_df = new_details_df

    # =========================
    # 5. 保存
    # =========================

    with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
        summary_df.to_excel(
            writer,
            sheet_name="summary_metrics",
            index=False
        )

        details_df.to_excel(
            writer,
            sheet_name="predicate_details",
            index=False
        )

    print(f"Saved experiment [{experiment_name}] to {output_path}")


append_result_to_excel(
    log_text=log_text,
    experiment_name=experiment_name,
    output_path=output_path
)