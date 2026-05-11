import json
import textwrap
from pathlib import Path


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip("\n").splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """
        # VG 评估结果综合分析 Notebook

        这个 notebook 用于系统分析 `vg_eval.py` 产出的评估结果文件，重点回答以下问题：

        - 模型整体表现如何，官方 `R@K / mR@K / ng-mR@K` 是多少
        - 哪些 GT 三元组没有被模型预测出来
        - 哪些三元组不在 GT 中，但模型却给出了预测
        - 对于某个谓词，不同 `(subject, predicate, object)` 组合之间的精度差异如何
        - 训练集里某个谓词出现得越多，模型对它的精确率是否越高
        - 哪些错误更偏向长尾、零样本或谓词混淆

        notebook 的说明文字、标题和关键代码注释均使用中文；函数和变量名保留英文，便于维护与复用。
        """
    ),
    markdown_cell(
        """
        ## 1. 环境与路径配置

        这一节负责：

        - 设置实验结果目录
        - 读取模型配置文件
        - 指定 Top-K、IoU 阈值和导出目录
        - 初始化绘图和表格展示风格

        如果你第一次使用这个 notebook，请优先修改 `RESULT_DIR` 和 `CONFIG_PATH`。
        """
    ),
    code_cell(
        '''
        import json
        import math
        import warnings
        from collections import Counter, defaultdict
        from pathlib import Path
        from typing import Dict, Iterable, List, Optional, Tuple

        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import Markdown, display
        from PIL import Image
        from tqdm.auto import tqdm

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            plt = None
            warnings.warn(f"未能导入 matplotlib，绘图相关单元将自动降级。详细原因: {exc}")

        try:
            import seaborn as sns
            if plt is not None:
                sns.set_theme(style="whitegrid", context="talk")
        except Exception:
            sns = None
            if plt is not None:
                plt.style.use("ggplot")

        pd.set_option("display.max_columns", 120)
        pd.set_option("display.max_rows", 200)
        pd.set_option("display.width", 200)

        REPO_ROOT = Path.cwd()
        if not (REPO_ROOT / "maskrcnn_benchmark").exists():
            for parent in REPO_ROOT.parents:
                if (parent / "maskrcnn_benchmark").exists():
                    REPO_ROOT = parent
                    break

        # 请在这里填写你的实验结果目录。该目录通常包含 eval_results.pytorch / result_dict.pytorch / visual_info.json。
        RESULT_DIR = REPO_ROOT / "output" / "please_set_result_dir"

        # 请在这里填写实验对应的配置文件。
        CONFIG_PATH = REPO_ROOT / "configs" / "e2e_relation_X_101_32_8_FPN_1x.yaml"

        # Top-K 设置尽量和官方评估保持一致。
        TOPK_LIST = [20, 50, 100]

        # 如果这里设置为 None，后面会优先尝试从配置中读取官方阈值；读不到时回退到 0.5。
        IOU_THRESH = None

        # 导出目录为空时默认使用 RESULT_DIR / "analysis_exports"。
        EXPORT_DIR = None

        # 每个谓词默认展示多少个代表性案例。
        SHOW_CASES_PER_PREDICATE = 3

        # 是否在案例可视化中显示图片。
        ENABLE_IMAGE_VIS = True

        print(f"仓库根目录: {REPO_ROOT}")
        print(f"结果目录: {RESULT_DIR}")
        print(f"配置文件: {CONFIG_PATH}")
        '''
    ),
    markdown_cell(
        """
        ## 2. 加载评估结果与类别映射

        这一节会读取：

        - `eval_results.pytorch`：包含 `groundtruths` 和 `predictions`
        - `result_dict.pytorch`：官方评估阶段累积出来的指标容器
        - `visual_info.json`：图片路径和基础可视化信息
        - VG 数据集字典文件：把对象类别 id 和谓词 id 转换成可读字符串
        - VG 训练集标注：统计每个谓词和每个三元组在训练集中的出现次数

        notebook 会对缺失文件做降级处理，并在文本提示中明确说明影响范围。
        """
    ),
    code_cell(
        '''
        def show_message(text: str) -> None:
            """用统一格式输出中文提示信息。"""
            display(Markdown(text))


        def ensure_path_exists(path: Path, description: str, required: bool = True) -> bool:
            """检查路径是否存在，并根据 required 决定是否抛出异常。"""
            if path.exists():
                return True
            message = f"{description}不存在: `{path}`"
            if required:
                raise FileNotFoundError(message)
            warnings.warn(message)
            return False


        def load_runtime_cfg(config_path: Path):
            """
            按训练脚本的方式构造运行时配置。

            这里显式复用 `maskrcnn_benchmark.config.cfg`，它本身来自
            `defaults.py`，因此会先带上全部默认参数，再和传入的
            config_file 做 merge，行为与 `tools/relation_train_net.py`
            保持一致。
            """
            ensure_path_exists(config_path, "配置文件")
            from maskrcnn_benchmark.config import cfg as global_cfg

            cfg_local = global_cfg.clone()
            cfg_local.merge_from_file(str(config_path))
            cfg_local.freeze()
            return cfg_local


        def normalize_dataset_names(dataset_value) -> List[str]:
            """
            把配置中的数据集字段统一规范成字符串列表。

            这里主要兼容两类写法：
            - 合法 YAML 列表，例如 ["VG_xxx_train"]
            - 项目里常见的 Python tuple 风格字符串，例如 ("VG_xxx_train",)
            """
            if dataset_value is None:
                return []
            if isinstance(dataset_value, (list, tuple)):
                return [str(item).strip() for item in dataset_value if str(item).strip()]
            if isinstance(dataset_value, str):
                text = dataset_value.strip()
                if not text:
                    return []
                if text.startswith("(") and text.endswith(")"):
                    text = text[1:-1].strip()
                if not text:
                    return []
                parts = [part.strip() for part in text.split(",")]
                cleaned = []
                for part in parts:
                    part = part.strip().strip('"').strip("'")
                    if part:
                        cleaned.append(part)
                return cleaned
            return [str(dataset_value).strip()]


        def resolve_dataset_paths(runtime_cfg) -> dict:
            """
            根据配置文件解析 VG 数据集路径。

            优先尝试复用仓库内部的 DatasetCatalog；如果失败，则回退到仓库默认 VG 路径。
            """
            dataset_names = normalize_dataset_names(runtime_cfg.DATASETS.TRAIN)
            train_dataset_name = dataset_names[0] if dataset_names else "VG_stanford_filtered_with_attribute_train"
            resolved = {
                "train_dataset_name": train_dataset_name,
                "dict_file": None,
                "roidb_file": None,
                "image_file": None,
                "img_dir": None,
            }

            try:
                from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog

                dataset_entry = DatasetCatalog.get(train_dataset_name, runtime_cfg)
                args = dataset_entry["args"]
                resolved.update({
                    "dict_file": Path(args["dict_file"]),
                    "roidb_file": Path(args["roidb_file"]),
                    "image_file": Path(args["image_file"]),
                    "img_dir": Path(args["img_dir"]),
                })
            except Exception as exc:
                warnings.warn(f"通过 DatasetCatalog 解析数据集路径失败，使用默认 VG 路径。详细原因: {exc}")
                dataset_root = REPO_ROOT / "datasets" / "vg"
                resolved.update({
                    "dict_file": dataset_root / "VG-SGG-dicts-with-attri.json",
                    "roidb_file": dataset_root / "VG-SGG-with-attri.h5",
                    "image_file": dataset_root / "image_data.json",
                    "img_dir": dataset_root / "VG_100K",
                })

            return resolved


        def resolve_iou_threshold(runtime_cfg, user_value: Optional[float]) -> float:
            """优先使用用户显式设置的阈值，否则从配置读取，最后回退到 0.5。"""
            if user_value is not None:
                return float(user_value)
            try:
                return float(runtime_cfg.TEST.RELATION.IOU_THRESHOLD)
            except Exception:
                return 0.5


        def load_label_mappings(dict_file: Path) -> dict:
            """从 VG 字典文件中读取对象类别和谓词类别映射。"""
            ensure_path_exists(dict_file, "VG 字典文件")
            with dict_file.open("r", encoding="utf-8") as f:
                info = json.load(f)

            label_to_idx = dict(info["label_to_idx"])
            predicate_to_idx = dict(info["predicate_to_idx"])
            attribute_to_idx = dict(info.get("attribute_to_idx", {}))

            label_to_idx["__background__"] = 0
            predicate_to_idx["__background__"] = 0
            attribute_to_idx["__background__"] = 0

            ind_to_classes = sorted(label_to_idx, key=lambda key: label_to_idx[key])
            ind_to_predicates = sorted(predicate_to_idx, key=lambda key: predicate_to_idx[key])
            ind_to_attributes = sorted(attribute_to_idx, key=lambda key: attribute_to_idx[key])

            return {
                "label_to_idx": label_to_idx,
                "predicate_to_idx": predicate_to_idx,
                "attribute_to_idx": attribute_to_idx,
                "ind_to_classes": ind_to_classes,
                "ind_to_predicates": ind_to_predicates,
                "ind_to_attributes": ind_to_attributes,
            }


        def empty_label_mappings() -> dict:
            """在字典文件不可用时返回空映射，避免 notebook 提前中断。"""
            return {
                "label_to_idx": {},
                "predicate_to_idx": {},
                "attribute_to_idx": {},
                "ind_to_classes": [],
                "ind_to_predicates": [],
                "ind_to_attributes": [],
            }


        def safe_name(index: int, name_list: List[str], unknown_prefix: str) -> str:
            """安全地把 id 转换成字符串，避免越界导致 notebook 中断。"""
            if 0 <= int(index) < len(name_list):
                return name_list[int(index)]
            return f"{unknown_prefix}_{index}"


        def load_eval_artifacts(result_dir: Path) -> dict:
            """读取评估输出文件，并对缺失的可选文件做降级处理。"""
            eval_path = result_dir / "eval_results.pytorch"
            result_dict_path = result_dir / "result_dict.pytorch"
            visual_info_path = result_dir / "visual_info.json"

            if not ensure_path_exists(result_dir, "结果目录", required=False):
                show_message("**提示：** 当前结果目录不存在，请先修改 `RESULT_DIR` 再重新运行本节。")
                return {}

            ensure_path_exists(eval_path, "eval_results.pytorch")
            eval_payload = torch.load(eval_path, map_location=torch.device("cpu"))
            result_dict = None
            visual_info = None

            if result_dict_path.exists():
                result_dict = torch.load(result_dict_path, map_location=torch.device("cpu"))
            else:
                warnings.warn("未找到 result_dict.pytorch，后续无法直接读取官方 mean recall 列表。")

            if visual_info_path.exists():
                with visual_info_path.open("r", encoding="utf-8") as f:
                    visual_info = json.load(f)
            else:
                warnings.warn("未找到 visual_info.json，后续案例展示将缺少图片路径信息。")

            return {
                "groundtruths": eval_payload["groundtruths"],
                "predictions": eval_payload["predictions"],
                "result_dict": result_dict,
                "visual_info": visual_info,
                "eval_path": eval_path,
                "result_dict_path": result_dict_path,
                "visual_info_path": visual_info_path,
            }


        def build_visual_lookup(visual_info: Optional[list]) -> dict:
            """把 visual_info 转成以 image_id 为键的索引字典。"""
            if not visual_info:
                return {}
            return {idx: item for idx, item in enumerate(visual_info)}


        def build_train_dataset_from_config(runtime_cfg):
            """
            按照真实训练配置构造训练数据集实例。

            这样可以确保以下过滤逻辑全部生效：
            - OV_SETTING 下的 base / novel / semantic / total 划分
            - 训练时的数据集 split 选择
            - 与 DatasetCatalog 一致的数据路径解析
            """
            from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
            from maskrcnn_benchmark.data import datasets as D

            dataset_names = normalize_dataset_names(runtime_cfg.DATASETS.TRAIN)
            if not dataset_names:
                raise ValueError("配置文件中的 DATASETS.TRAIN 为空，无法构造训练集。")

            train_dataset_name = dataset_names[0]
            dataset_entry = DatasetCatalog.get(train_dataset_name, runtime_cfg)
            factory = getattr(D, dataset_entry["factory"])
            args = dict(dataset_entry["args"])
            if "capgraphs_file" in args:
                del args["capgraphs_file"]
            args["transforms"] = None
            return factory(**args)


        def compute_training_statistics(runtime_cfg, ind_to_classes: List[str], ind_to_predicates: List[str]) -> dict:
            """
            统计训练集中每个谓词和每个三元组的出现次数。

            这里必须按“真实训练数据集”统计，而不是直接读取全量 VG 原始标注。
            原因是项目里的 VGDataset 会在实例化时根据 OV_SETTING 对 base / novel / semantic
            词表做过滤；如果绕过这一步，统计出来的频次会包含训练时根本没见过的关系。
            """
            try:
                train_dataset = build_train_dataset_from_config(runtime_cfg)
            except Exception as exc:
                warnings.warn(f"无法按真实配置构造训练数据集，训练频次统计将不可用。详细原因: {exc}")
                return {
                    "predicate_counter": Counter(),
                    "triplet_counter": Counter(),
                    "triplet_name_counter": Counter(),
                }

            predicate_counter = Counter()
            triplet_counter = Counter()
            triplet_name_counter = Counter()

            for classes_per_image, rels_per_image in tqdm(
                zip(train_dataset.gt_classes, train_dataset.relationships),
                total=len(train_dataset.relationships),
                desc="统计训练集谓词与三元组频次",
            ):
                for sub_idx, obj_idx, predicate_id in rels_per_image:
                    predicate_id = int(predicate_id)
                    sub_label = int(classes_per_image[int(sub_idx)])
                    obj_label = int(classes_per_image[int(obj_idx)])
                    triplet_key = (sub_label, predicate_id, obj_label)
                    triplet_name = (
                        safe_name(sub_label, ind_to_classes, "unknown_obj"),
                        safe_name(predicate_id, ind_to_predicates, "unknown_pred"),
                        safe_name(obj_label, ind_to_classes, "unknown_obj"),
                    )
                    predicate_counter[predicate_id] += 1
                    triplet_counter[triplet_key] += 1
                    triplet_name_counter[triplet_name] += 1

            return {
                "predicate_counter": predicate_counter,
                "triplet_counter": triplet_counter,
                "triplet_name_counter": triplet_name_counter,
            }


        runtime_cfg = load_runtime_cfg(CONFIG_PATH)
        dataset_paths = resolve_dataset_paths(runtime_cfg)
        iou_threshold = resolve_iou_threshold(runtime_cfg, IOU_THRESH)
        dict_file = dataset_paths.get("dict_file")
        if dict_file and Path(dict_file).exists():
            label_mapping = load_label_mappings(Path(dict_file))
        else:
            warnings.warn("当前未找到 VG 字典文件；如果你还没有填写正确的 RESULT_DIR，这通常是正常现象。")
            label_mapping = empty_label_mappings()
        artifacts = load_eval_artifacts(RESULT_DIR)

        context = None
        if artifacts:
            if not label_mapping["ind_to_classes"] or not label_mapping["ind_to_predicates"]:
                raise FileNotFoundError("结果文件已找到，但 VG 字典文件缺失，无法把类别 id 转成可读字符串。请检查 CONFIG_PATH 和数据集路径。")

            train_statistics = compute_training_statistics(
                runtime_cfg=runtime_cfg,
                ind_to_classes=label_mapping["ind_to_classes"],
                ind_to_predicates=label_mapping["ind_to_predicates"],
            )

            zeroshot_path = REPO_ROOT / "maskrcnn_benchmark" / "data" / "datasets" / "evaluation" / "vg" / "zeroshot_triplet.pytorch"
            zeroshot_triplets = set()
            if zeroshot_path.exists():
                zeroshot_triplets = {
                    tuple(map(int, row))
                    for row in torch.load(zeroshot_path, map_location=torch.device("cpu")).long().numpy()
                }

            export_dir = Path(EXPORT_DIR) if EXPORT_DIR else RESULT_DIR / "analysis_exports"
            context = {
                "dataset_paths": dataset_paths,
                "runtime_cfg": runtime_cfg,
                "label_mapping": label_mapping,
                "artifacts": artifacts,
                "train_statistics": train_statistics,
                "visual_lookup": build_visual_lookup(artifacts["visual_info"]),
                "iou_threshold": iou_threshold,
                "topk_list": sorted(set(int(v) for v in TOPK_LIST)),
                "zeroshot_triplets": zeroshot_triplets,
                "export_dir": export_dir,
            }

            show_message(
                f"""
                **加载完成**

                - 图片数量：`{len(artifacts["groundtruths"])}`  
                - GT / 预测文件：`{artifacts["eval_path"]}`  
                - IoU 阈值：`{iou_threshold:.2f}`  
                - 训练集谓词种类数（含背景）：`{len(label_mapping["ind_to_predicates"])}`  
                - 导出目录：`{export_dir}`
                """
            )
        '''
    ),
    markdown_cell(
        """
        ## 3. 还原官方评估口径

        下面这组函数会复现 `vg_eval.py / sgg_eval.py` 的核心逻辑：

        - GT 三元组：`(subject_class, predicate, object_class)`
        - 预测三元组：`(subject_class, predicate_argmax, object_class)`
        - 匹配条件：三元组标签完全一致，且主语框和宾语框的 IoU 同时不低于阈值

        为了保证和官方结果尽量一致：

        - graph-constrained 分析保留 `eval_results.pytorch` 中原始关系顺序
        - no-graph-constraint 分析会重建所有 `(pair, predicate)` 候选并按综合分数排序
        - 所有统计都保留 `count / hit / precision / recall` 等基础计数，便于复查
        """
    ),
    code_cell(
        '''
        def compute_box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
            """计算两组框之间的 IoU 矩阵。"""
            if boxes1.size == 0 or boxes2.size == 0:
                return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

            boxes1 = boxes1.astype(np.float32)
            boxes2 = boxes2.astype(np.float32)

            area1 = np.clip(boxes1[:, 2] - boxes1[:, 0] + 1, a_min=0, a_max=None) * np.clip(boxes1[:, 3] - boxes1[:, 1] + 1, a_min=0, a_max=None)
            area2 = np.clip(boxes2[:, 2] - boxes2[:, 0] + 1, a_min=0, a_max=None) * np.clip(boxes2[:, 3] - boxes2[:, 1] + 1, a_min=0, a_max=None)

            lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
            rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
            wh = np.clip(rb - lt + 1, a_min=0, a_max=None)
            inter = wh[:, :, 0] * wh[:, :, 1]
            union = area1[:, None] + area2[None, :] - inter
            return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


        def intersect_2d_numpy(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
            """判断 arr1 中每一行是否与 arr2 中每一行完全相等。"""
            if arr1.size == 0 or arr2.size == 0:
                return np.zeros((len(arr1), len(arr2)), dtype=bool)
            return (arr1[:, None, :] == arr2[None, :, :]).all(axis=-1)


        def build_triplets(relations: np.ndarray, classes: np.ndarray, boxes: np.ndarray, predicate_scores: Optional[np.ndarray] = None, class_scores: Optional[np.ndarray] = None):
            """把 (sub_idx, obj_idx, pred_id) 转成 triplet 形式。"""
            if len(relations) == 0:
                empty_triplets = np.zeros((0, 3), dtype=np.int64)
                empty_boxes = np.zeros((0, 8), dtype=np.float32)
                empty_scores = np.zeros((0, 3), dtype=np.float32) if predicate_scores is not None and class_scores is not None else None
                return empty_triplets, empty_boxes, empty_scores

            sub_id = relations[:, 0].astype(np.int64)
            obj_id = relations[:, 1].astype(np.int64)
            pred_label = relations[:, 2].astype(np.int64)

            triplets = np.column_stack((classes[sub_id], pred_label, classes[obj_id]))
            triplet_boxes = np.column_stack((boxes[sub_id], boxes[obj_id])).astype(np.float32)

            triplet_scores = None
            if predicate_scores is not None and class_scores is not None:
                triplet_scores = np.column_stack((
                    class_scores[sub_id],
                    predicate_scores,
                    class_scores[obj_id],
                )).astype(np.float32)

            return triplets, triplet_boxes, triplet_scores


        def compute_pred_matches(gt_triplets: np.ndarray, pred_triplets: np.ndarray, gt_triplet_boxes: np.ndarray, pred_triplet_boxes: np.ndarray, iou_thresh: float) -> List[List[int]]:
            """
            复现官方评估中的 GT-预测匹配关系。

            返回值 pred_to_gt 的长度等于预测关系数，其中每个元素是当前预测命中的 GT 索引列表。
            """
            keeps = intersect_2d_numpy(gt_triplets, pred_triplets)
            gt_has_match = keeps.any(axis=1)
            pred_to_gt = [[] for _ in range(len(pred_triplets))]

            for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0], gt_triplet_boxes[gt_has_match], keeps[gt_has_match]):
                pred_indices = np.where(keep_inds)[0]
                if len(pred_indices) == 0:
                    continue
                boxes = pred_triplet_boxes[pred_indices]
                gt_sub_box = gt_box[:4][None, :]
                gt_obj_box = gt_box[4:][None, :]
                pred_sub_boxes = boxes[:, :4]
                pred_obj_boxes = boxes[:, 4:]
                sub_iou = compute_box_iou_matrix(gt_sub_box, pred_sub_boxes)[0]
                obj_iou = compute_box_iou_matrix(gt_obj_box, pred_obj_boxes)[0]
                matched_pred_indices = pred_indices[(sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)]
                for pred_index in matched_pred_indices.tolist():
                    pred_to_gt[pred_index].append(int(gt_ind))

            return pred_to_gt


        def flatten_top_scores(score_matrix: np.ndarray, limit: int) -> np.ndarray:
            """对二维分数矩阵做降序展开，返回前 limit 个位置的二维索引。"""
            if score_matrix.size == 0:
                return np.zeros((0, 2), dtype=np.int64)
            flat_indices = np.argsort(score_matrix.reshape(-1))[::-1]
            flat_indices = flat_indices[:limit]
            return np.column_stack(np.unravel_index(flat_indices, score_matrix.shape))


        def build_nogc_predictions(pred_rel_inds: np.ndarray, rel_scores: np.ndarray, obj_scores: np.ndarray, top_limit: int) -> dict:
            """复现 no-graph-constraint 评估时的候选排序。"""
            if len(pred_rel_inds) == 0:
                return {
                    "pred_rels": np.zeros((0, 3), dtype=np.int64),
                    "predicate_scores": np.zeros((0,), dtype=np.float32),
                    "triplet_scores": np.zeros((0,), dtype=np.float32),
                    "pair_rank_source": np.zeros((0,), dtype=np.int64),
                }

            obj_scores_per_rel = obj_scores[pred_rel_inds].prod(axis=1)
            overall_scores = obj_scores_per_rel[:, None] * rel_scores[:, 1:]
            top_indices = flatten_top_scores(overall_scores, top_limit)

            pred_rels = np.column_stack((
                pred_rel_inds[top_indices[:, 0]],
                top_indices[:, 1] + 1,
            )).astype(np.int64)
            predicate_scores = rel_scores[top_indices[:, 0], top_indices[:, 1] + 1].astype(np.float32)
            triplet_scores = overall_scores[top_indices[:, 0], top_indices[:, 1]].astype(np.float32)
            pair_rank_source = top_indices[:, 0].astype(np.int64)

            return {
                "pred_rels": pred_rels,
                "predicate_scores": predicate_scores,
                "triplet_scores": triplet_scores,
                "pair_rank_source": pair_rank_source,
            }


        def union_pred_to_gt(pred_to_gt: List[List[int]], topk: int) -> List[int]:
            """把前 topk 个预测命中的 GT 索引做并集。"""
            if topk <= 0 or len(pred_to_gt) == 0:
                return []
            merged = set()
            for matched_gt in pred_to_gt[: min(topk, len(pred_to_gt))]:
                merged.update(int(item) for item in matched_gt)
            return sorted(merged)
        '''
    ),
    code_cell(
        '''
        def analyze_one_image(
            image_id: int,
            groundtruth,
            prediction,
            visual_lookup: dict,
            ind_to_classes: List[str],
            ind_to_predicates: List[str],
            iou_thresh: float,
            topk_list: List[int],
        ) -> dict:
            """
            对单张图片做关系级分析，输出 GT 记录、graph-constrained 预测记录、no-graph 预测记录和谓词混淆线索。
            """
            gt_boxes = groundtruth.convert("xyxy").bbox.detach().cpu().numpy()
            gt_classes = groundtruth.get_field("labels").long().detach().cpu().numpy()
            gt_rels = groundtruth.get_field("relation_tuple").long().detach().cpu().numpy()

            pred_boxes = prediction.convert("xyxy").bbox.detach().cpu().numpy()
            pred_classes = prediction.get_field("pred_labels").long().detach().cpu().numpy()
            obj_scores = prediction.get_field("pred_scores").detach().cpu().numpy()
            pred_rel_inds = prediction.get_field("rel_pair_idxs").long().detach().cpu().numpy()
            rel_scores = prediction.get_field("pred_rel_scores").detach().cpu().numpy()

            max_topk = max(topk_list) if topk_list else 100
            image_info = visual_lookup.get(image_id, {})
            img_file = image_info.get("img_file")

            gt_triplets, gt_triplet_boxes, _ = build_triplets(gt_rels, gt_classes, gt_boxes)

            if len(pred_rel_inds) > 0:
                gc_predicates = 1 + rel_scores[:, 1:].argmax(axis=1)
                gc_predicate_scores = rel_scores[:, 1:].max(axis=1)
                gc_pred_rels = np.column_stack((pred_rel_inds, gc_predicates)).astype(np.int64)
                gc_triplets, gc_triplet_boxes, gc_triplet_scores_3 = build_triplets(
                    gc_pred_rels,
                    pred_classes,
                    pred_boxes,
                    predicate_scores=gc_predicate_scores,
                    class_scores=obj_scores,
                )
                gc_triplet_scores = gc_triplet_scores_3.prod(axis=1) if len(gc_triplet_scores_3) > 0 else np.zeros((0,), dtype=np.float32)
                gc_pred_to_gt = compute_pred_matches(
                    gt_triplets=gt_triplets,
                    pred_triplets=gc_triplets,
                    gt_triplet_boxes=gt_triplet_boxes,
                    pred_triplet_boxes=gc_triplet_boxes,
                    iou_thresh=iou_thresh,
                )
            else:
                gc_pred_rels = np.zeros((0, 3), dtype=np.int64)
                gc_predicate_scores = np.zeros((0,), dtype=np.float32)
                gc_triplets = np.zeros((0, 3), dtype=np.int64)
                gc_triplet_boxes = np.zeros((0, 8), dtype=np.float32)
                gc_triplet_scores = np.zeros((0,), dtype=np.float32)
                gc_pred_to_gt = []

            nogc_payload = build_nogc_predictions(pred_rel_inds, rel_scores, obj_scores, top_limit=max(100, max_topk))
            nogc_pred_rels = nogc_payload["pred_rels"]
            nogc_predicate_scores = nogc_payload["predicate_scores"]
            nogc_triplet_scores = nogc_payload["triplet_scores"]

            if len(nogc_pred_rels) > 0:
                nogc_triplets, nogc_triplet_boxes, _ = build_triplets(
                    nogc_pred_rels,
                    pred_classes,
                    pred_boxes,
                    predicate_scores=nogc_predicate_scores,
                    class_scores=obj_scores,
                )
                nogc_pred_to_gt = compute_pred_matches(
                    gt_triplets=gt_triplets,
                    pred_triplets=nogc_triplets,
                    gt_triplet_boxes=gt_triplet_boxes,
                    pred_triplet_boxes=nogc_triplet_boxes,
                    iou_thresh=iou_thresh,
                )
            else:
                nogc_triplets = np.zeros((0, 3), dtype=np.int64)
                nogc_triplet_boxes = np.zeros((0, 8), dtype=np.float32)
                nogc_pred_to_gt = []

            matched_gt_per_k = {k: set(union_pred_to_gt(gc_pred_to_gt, k)) for k in topk_list}
            matched_gt_nogc_per_k = {k: set(union_pred_to_gt(nogc_pred_to_gt, k)) for k in topk_list}

            gt_rows = []
            confusion_rows = []
            for gt_index, (sub_idx, obj_idx, predicate_id) in enumerate(gt_rels):
                sub_idx = int(sub_idx)
                obj_idx = int(obj_idx)
                predicate_id = int(predicate_id)
                subject_id = int(gt_classes[sub_idx])
                object_id = int(gt_classes[obj_idx])
                triplet_key = (subject_id, predicate_id, object_id)
                triplet_str = f"{safe_name(subject_id, ind_to_classes, 'unknown_obj')} - {safe_name(predicate_id, ind_to_predicates, 'unknown_pred')} - {safe_name(object_id, ind_to_classes, 'unknown_obj')}"

                row = {
                    "image_id": image_id,
                    "img_file": img_file,
                    "gt_index": gt_index,
                    "subject_box_index": sub_idx,
                    "object_box_index": obj_idx,
                    "subject_id": subject_id,
                    "predicate_id": predicate_id,
                    "object_id": object_id,
                    "subject_name": safe_name(subject_id, ind_to_classes, "unknown_obj"),
                    "predicate_name": safe_name(predicate_id, ind_to_predicates, "unknown_pred"),
                    "object_name": safe_name(object_id, ind_to_classes, "unknown_obj"),
                    "triplet_key": triplet_key,
                    "triplet_str": triplet_str,
                    "gt_sub_box": gt_boxes[sub_idx].tolist(),
                    "gt_obj_box": gt_boxes[obj_idx].tolist(),
                }

                for k in topk_list:
                    row[f"matched_at_{k}"] = gt_index in matched_gt_per_k[k]
                    row[f"matched_nogc_at_{k}"] = gt_index in matched_gt_nogc_per_k[k]

                best_candidate = None
                if len(gc_pred_rels) > 0:
                    candidate_indices = []
                    for pred_index, pred_rel in enumerate(gc_pred_rels):
                        pred_sub_idx, pred_obj_idx, pred_predicate = [int(v) for v in pred_rel]
                        pred_subject_id = int(pred_classes[pred_sub_idx])
                        pred_object_id = int(pred_classes[pred_obj_idx])
                        if pred_subject_id != subject_id or pred_object_id != object_id:
                            continue
                        subject_iou = float(compute_box_iou_matrix(gt_boxes[sub_idx][None, :], pred_boxes[pred_sub_idx][None, :])[0, 0])
                        object_iou = float(compute_box_iou_matrix(gt_boxes[obj_idx][None, :], pred_boxes[pred_obj_idx][None, :])[0, 0])
                        if subject_iou >= iou_thresh and object_iou >= iou_thresh:
                            candidate_indices.append((pred_index, subject_iou, object_iou))

                    if candidate_indices:
                        pred_index, subject_iou, object_iou = sorted(
                            candidate_indices,
                            key=lambda item: gc_triplet_scores[item[0]],
                            reverse=True,
                        )[0]
                        candidate_predicate = int(gc_pred_rels[pred_index][2])
                        best_candidate = {
                            "candidate_pred_index": pred_index,
                            "candidate_predicate_id": candidate_predicate,
                            "candidate_predicate_name": safe_name(candidate_predicate, ind_to_predicates, "unknown_pred"),
                            "candidate_triplet_score": float(gc_triplet_scores[pred_index]),
                            "candidate_predicate_score": float(gc_predicate_scores[pred_index]),
                            "candidate_subject_iou": subject_iou,
                            "candidate_object_iou": object_iou,
                        }

                confusion_rows.append({
                    "image_id": image_id,
                    "img_file": img_file,
                    "gt_index": gt_index,
                    "gt_predicate_id": predicate_id,
                    "gt_predicate_name": safe_name(predicate_id, ind_to_predicates, "unknown_pred"),
                    "gt_triplet_str": triplet_str,
                    "matched_at_100": row.get("matched_at_100", False),
                    "candidate_predicate_id": best_candidate["candidate_predicate_id"] if best_candidate else -1,
                    "candidate_predicate_name": best_candidate["candidate_predicate_name"] if best_candidate else "__no_prediction__",
                    "candidate_triplet_score": best_candidate["candidate_triplet_score"] if best_candidate else np.nan,
                })
                gt_rows.append(row)

            pred_rows = []
            for pred_index, pred_rel in enumerate(gc_pred_rels):
                pred_sub_idx, pred_obj_idx, predicate_id = [int(v) for v in pred_rel]
                subject_id = int(pred_classes[pred_sub_idx])
                object_id = int(pred_classes[pred_obj_idx])
                triplet_key = (subject_id, predicate_id, object_id)
                row = {
                    "image_id": image_id,
                    "img_file": img_file,
                    "pred_index": pred_index,
                    "subject_box_index": pred_sub_idx,
                    "object_box_index": pred_obj_idx,
                    "subject_id": subject_id,
                    "predicate_id": predicate_id,
                    "object_id": object_id,
                    "subject_name": safe_name(subject_id, ind_to_classes, "unknown_obj"),
                    "predicate_name": safe_name(predicate_id, ind_to_predicates, "unknown_pred"),
                    "object_name": safe_name(object_id, ind_to_classes, "unknown_obj"),
                    "triplet_key": triplet_key,
                    "triplet_str": f"{safe_name(subject_id, ind_to_classes, 'unknown_obj')} - {safe_name(predicate_id, ind_to_predicates, 'unknown_pred')} - {safe_name(object_id, ind_to_classes, 'unknown_obj')}",
                    "triplet_score": float(gc_triplet_scores[pred_index]),
                    "predicate_score": float(gc_predicate_scores[pred_index]),
                    "matched_gt_indices": [int(v) for v in gc_pred_to_gt[pred_index]],
                }
                for k in topk_list:
                    row[f"is_top_{k}"] = pred_index < k
                    row[f"correct_at_{k}"] = pred_index < k and len(gc_pred_to_gt[pred_index]) > 0
                pred_rows.append(row)

            nogc_rows = []
            for pred_index, pred_rel in enumerate(nogc_pred_rels):
                pred_sub_idx, pred_obj_idx, predicate_id = [int(v) for v in pred_rel]
                subject_id = int(pred_classes[pred_sub_idx])
                object_id = int(pred_classes[pred_obj_idx])
                triplet_key = (subject_id, predicate_id, object_id)
                row = {
                    "image_id": image_id,
                    "img_file": img_file,
                    "pred_index": pred_index,
                    "subject_box_index": pred_sub_idx,
                    "object_box_index": pred_obj_idx,
                    "subject_id": subject_id,
                    "predicate_id": predicate_id,
                    "object_id": object_id,
                    "subject_name": safe_name(subject_id, ind_to_classes, "unknown_obj"),
                    "predicate_name": safe_name(predicate_id, ind_to_predicates, "unknown_pred"),
                    "object_name": safe_name(object_id, ind_to_classes, "unknown_obj"),
                    "triplet_key": triplet_key,
                    "triplet_str": f"{safe_name(subject_id, ind_to_classes, 'unknown_obj')} - {safe_name(predicate_id, ind_to_predicates, 'unknown_pred')} - {safe_name(object_id, ind_to_classes, 'unknown_obj')}",
                    "triplet_score": float(nogc_triplet_scores[pred_index]),
                    "predicate_score": float(nogc_predicate_scores[pred_index]),
                    "pair_rank_source": int(nogc_payload["pair_rank_source"][pred_index]),
                    "matched_gt_indices": [int(v) for v in nogc_pred_to_gt[pred_index]],
                }
                for k in topk_list:
                    row[f"is_top_{k}"] = pred_index < k
                    row[f"correct_at_{k}"] = pred_index < k and len(nogc_pred_to_gt[pred_index]) > 0
                nogc_rows.append(row)

            return {
                "gt_rows": gt_rows,
                "pred_rows": pred_rows,
                "nogc_rows": nogc_rows,
                "confusion_rows": confusion_rows,
            }


        if context is not None:
            analysis_rows = {
                "gt_rows": [],
                "pred_rows": [],
                "nogc_rows": [],
                "confusion_rows": [],
            }

            for image_id, (groundtruth, prediction) in tqdm(
                enumerate(zip(context["artifacts"]["groundtruths"], context["artifacts"]["predictions"])),
                total=len(context["artifacts"]["groundtruths"]),
                desc="逐图构建分析表",
            ):
                analyzed = analyze_one_image(
                    image_id=image_id,
                    groundtruth=groundtruth,
                    prediction=prediction,
                    visual_lookup=context["visual_lookup"],
                    ind_to_classes=context["label_mapping"]["ind_to_classes"],
                    ind_to_predicates=context["label_mapping"]["ind_to_predicates"],
                    iou_thresh=context["iou_threshold"],
                    topk_list=context["topk_list"],
                )
                for key in analysis_rows:
                    analysis_rows[key].extend(analyzed[key])

            gt_df = pd.DataFrame(analysis_rows["gt_rows"])
            pred_df = pd.DataFrame(analysis_rows["pred_rows"])
            nogc_pred_df = pd.DataFrame(analysis_rows["nogc_rows"])
            confusion_df = pd.DataFrame(analysis_rows["confusion_rows"])

            context["tables"] = {
                "gt_df": gt_df,
                "pred_df": pred_df,
                "nogc_pred_df": nogc_pred_df,
                "confusion_df": confusion_df,
            }

            show_message(
                f"""
                **分析表构建完成**

                - GT 关系数：`{len(gt_df)}`  
                - graph-constrained 预测关系数：`{len(pred_df)}`  
                - no-graph 预测关系数：`{len(nogc_pred_df)}`  
                - 谓词混淆线索数：`{len(confusion_df)}`
                """
            )
        '''
    ),
    markdown_cell(
        """
        ## 4. 全局指标汇总

        这里优先展示官方 `result_dict.pytorch` 中已经保存的结果，因为它和训练/测试脚本完全同口径。

        如果缺少 `result_dict.pytorch`，则 notebook 会退化为展示基于当前分析表重新聚合得到的近似指标，并明确说明与官方结果的差异。
        """
    ),
    code_cell(
        '''
        def detect_eval_mode(result_dict: Optional[dict]) -> Optional[str]:
            """从官方 result_dict 中推断当前评估模式，例如 predcls / sgcls / sgdet。"""
            if not result_dict:
                return None
            for key in result_dict:
                if key.endswith("_recall") and not key.endswith("_zeroshot_recall"):
                    return key.replace("_recall", "")
            return None


        def summarize_official_metrics(result_dict: Optional[dict]) -> pd.DataFrame:
            """把官方 result_dict 整理成更容易阅读的长表。"""
            if not result_dict:
                return pd.DataFrame()

            mode = detect_eval_mode(result_dict)
            if mode is None:
                return pd.DataFrame()

            metric_rows = []
            for metric_name, metric_value in result_dict.items():
                if not metric_name.startswith(mode):
                    continue
                suffix = metric_name.replace(f"{mode}_", "")
                if isinstance(metric_value, dict):
                    for k, values in metric_value.items():
                        scalar = safe_metric_to_scalar(values)
                        if pd.isna(scalar):
                            continue
                        metric_rows.append({
                            "mode": mode,
                            "metric_name": suffix,
                            "k": int(k),
                            "value": scalar,
                        })
            return pd.DataFrame(metric_rows).sort_values(["metric_name", "k"]).reset_index(drop=True)


        def safe_metric_to_scalar(values) -> float:
            """
            尝试把官方 result_dict 中的指标项转换成单个浮点数。

            只接受以下两类值：
            - 单个标量
            - 一维数值列表

            对于 `mean_recall_collect` 这类嵌套列表结构，直接返回 NaN 并跳过，
            避免 notebook 在展示官方指标时中断。
            """
            if isinstance(values, (int, float, np.integer, np.floating)):
                return float(values)

            if isinstance(values, np.ndarray):
                if values.ndim == 0:
                    return float(values.item())
                if values.ndim == 1 and np.issubdtype(values.dtype, np.number):
                    return float(np.mean(values)) if len(values) > 0 else np.nan
                return np.nan

            if isinstance(values, list):
                if len(values) == 0:
                    return np.nan
                if all(isinstance(item, (int, float, np.integer, np.floating)) for item in values):
                    return float(np.mean(values))
                return np.nan

            try:
                return float(values)
            except Exception:
                return np.nan


        def summarize_notebook_recall(gt_df: pd.DataFrame, pred_df: pd.DataFrame, topk_list: List[int]) -> pd.DataFrame:
            """在 result_dict 缺失时，给出一个便于参考的数据集级近似汇总。"""
            rows = []
            total_gt = len(gt_df)
            total_images = gt_df["image_id"].nunique() if len(gt_df) > 0 else 0

            for k in topk_list:
                matched_count = int(gt_df[f"matched_at_{k}"].sum()) if total_gt > 0 else 0
                pred_topk_count = int(pred_df[f"is_top_{k}"].sum()) if len(pred_df) > 0 else 0
                tp_count = int(pred_df[f"correct_at_{k}"].sum()) if len(pred_df) > 0 else 0
                rows.append({
                    "mode": "notebook",
                    "metric_name": "dataset_level_recall",
                    "k": k,
                    "value": matched_count / total_gt if total_gt > 0 else np.nan,
                })
                rows.append({
                    "mode": "notebook",
                    "metric_name": "dataset_level_precision",
                    "k": k,
                    "value": tp_count / pred_topk_count if pred_topk_count > 0 else np.nan,
                })
            if total_images > 0:
                for k in topk_list:
                    image_recall = gt_df.groupby("image_id")[f"matched_at_{k}"].mean().mean()
                    rows.append({
                        "mode": "notebook",
                        "metric_name": "image_average_gt_hit_rate",
                        "k": k,
                        "value": float(image_recall),
                    })
            return pd.DataFrame(rows)


        if context is not None:
            official_metrics_df = summarize_official_metrics(context["artifacts"]["result_dict"])
            notebook_metrics_df = summarize_notebook_recall(
                gt_df=context["tables"]["gt_df"],
                pred_df=context["tables"]["pred_df"],
                topk_list=context["topk_list"],
            )
            context["tables"]["official_metrics_df"] = official_metrics_df
            context["tables"]["notebook_metrics_df"] = notebook_metrics_df

            if len(official_metrics_df) > 0:
                show_message("### 官方结果概览")
                display(official_metrics_df.pivot_table(index="metric_name", columns="k", values="value", aggfunc="first"))
            else:
                show_message("### 未检测到官方 `result_dict.pytorch`，下面展示 notebook 复算的近似汇总")
                display(notebook_metrics_df.pivot_table(index="metric_name", columns="k", values="value", aggfunc="first"))
        '''
    ),
    markdown_cell(
        """
        ## 5. GT 漏检分析

        这一节聚焦“GT 中存在，但模型在 Top-K 内没有命中”的关系。

        输出包括：

        - 漏检最多的谓词
        - 漏检最多的三元组
        - 每张图片的漏检数量
        """
    ),
    code_cell(
        '''
        if context is not None:
            gt_df = context["tables"]["gt_df"].copy()
            missed_by_k = {}

            for k in context["topk_list"]:
                missed_df = gt_df.loc[~gt_df[f"matched_at_{k}"]].copy()
                missed_df["analysis_topk"] = k
                missed_by_k[k] = missed_df
                context["tables"][f"missed_gt_top{k}_df"] = missed_df

                show_message(f"### Top-{k} 下的 GT 漏检概览")
                print(f"漏检数量: {len(missed_df)} / {len(gt_df)}")

                predicate_view = (
                    missed_df.groupby(["predicate_id", "predicate_name"], as_index=False)
                    .size()
                    .rename(columns={"size": "missed_count"})
                    .sort_values("missed_count", ascending=False)
                    .head(20)
                )
                display(predicate_view)

                triplet_view = (
                    missed_df.groupby(["subject_name", "predicate_name", "object_name", "triplet_str"], as_index=False)
                    .size()
                    .rename(columns={"size": "missed_count"})
                    .sort_values("missed_count", ascending=False)
                    .head(20)
                )
                display(triplet_view)

                image_view = (
                    missed_df.groupby(["image_id", "img_file"], as_index=False)
                    .size()
                    .rename(columns={"size": "missed_count"})
                    .sort_values("missed_count", ascending=False)
                    .head(20)
                )
                display(image_view)
        '''
    ),
    markdown_cell(
        """
        ## 6. 非 GT 预测分析（伪阳性）

        这一节聚焦“模型给出了预测，但这些三元组没有命中任何 GT”的情况。

        默认展示 graph-constrained 视角下的 Top-K 伪阳性，也会保留 no-graph 预测表，便于需要时做扩展分析。
        """
    ),
    code_cell(
        '''
        if context is not None:
            pred_df = context["tables"]["pred_df"].copy()

            for k in context["topk_list"]:
                fp_df = pred_df.loc[pred_df[f"is_top_{k}"] & ~pred_df[f"correct_at_{k}"]].copy()
                fp_df["analysis_topk"] = k
                context["tables"][f"false_positive_top{k}_df"] = fp_df

                show_message(f"### Top-{k} 下的伪阳性概览")
                print(f"伪阳性数量: {len(fp_df)}")

                predicate_view = (
                    fp_df.groupby(["predicate_id", "predicate_name"], as_index=False)
                    .agg(
                        false_positive_count=("pred_index", "size"),
                        avg_triplet_score=("triplet_score", "mean"),
                        median_triplet_score=("triplet_score", "median"),
                    )
                    .sort_values("false_positive_count", ascending=False)
                    .head(20)
                )
                display(predicate_view)

                triplet_view = (
                    fp_df.groupby(["subject_name", "predicate_name", "object_name", "triplet_str"], as_index=False)
                    .agg(
                        false_positive_count=("pred_index", "size"),
                        avg_triplet_score=("triplet_score", "mean"),
                    )
                    .sort_values(["false_positive_count", "avg_triplet_score"], ascending=[False, False])
                    .head(20)
                )
                display(triplet_view)

                score_bins = pd.cut(
                    fp_df["triplet_score"],
                    bins=[-np.inf, 0.01, 0.05, 0.1, 0.2, 0.5, np.inf],
                    labels=["<=0.01", "(0.01,0.05]", "(0.05,0.1]", "(0.1,0.2]", "(0.2,0.5]", ">0.5"],
                )
                score_view = score_bins.value_counts(dropna=False).rename_axis("score_bin").reset_index(name="false_positive_count")
                display(score_view)
        '''
    ),
    markdown_cell(
        """
        ## 7. 谓词级统计分析

        这一节是整个 notebook 的核心之一，会为每个谓词提供：

        - 测试集 GT 数量
        - Top-K 命中数量
        - Top-K 预测数量
        - 精确率 / 召回率 / F1
        - 官方 mean recall（若 `result_dict.pytorch` 可用）
        - 训练集中的谓词出现次数
        - `precision / log(train_count + 1)` 这类频次归一化比值
        """
    ),
    code_cell(
        '''
        def get_official_mean_recall_map(result_dict: Optional[dict]) -> dict:
            """提取官方 result_dict 中每个谓词的 mean recall 列表。"""
            mode = detect_eval_mode(result_dict)
            if not result_dict or mode is None:
                return {}
            key = f"{mode}_mean_recall_list"
            if key not in result_dict or 100 not in result_dict[key]:
                return {}
            values = result_dict[key][100]
            return {index + 1: float(value) for index, value in enumerate(values)}


        def compute_predicate_metrics(
            gt_df: pd.DataFrame,
            pred_df: pd.DataFrame,
            label_mapping: dict,
            topk_list: List[int],
            train_statistics: dict,
            result_dict: Optional[dict],
        ) -> pd.DataFrame:
            """聚合每个谓词在测试集上的命中、预测、精确率和召回率。"""
            predicate_ids = [idx for idx, name in enumerate(label_mapping["ind_to_predicates"]) if idx != 0]
            official_mean_recall_map = get_official_mean_recall_map(result_dict)

            rows = []
            for predicate_id in predicate_ids:
                predicate_name = safe_name(predicate_id, label_mapping["ind_to_predicates"], "unknown_pred")
                gt_slice = gt_df.loc[gt_df["predicate_id"] == predicate_id]
                pred_slice = pred_df.loc[pred_df["predicate_id"] == predicate_id]
                base_row = {
                    "predicate_id": predicate_id,
                    "predicate_name": predicate_name,
                    "gt_count": int(len(gt_slice)),
                    "train_count": int(train_statistics["predicate_counter"].get(predicate_id, 0)),
                    "official_mean_recall_at_100": official_mean_recall_map.get(predicate_id, np.nan),
                }

                for k in topk_list:
                    hit_count = int(gt_slice[f"matched_at_{k}"].sum()) if len(gt_slice) > 0 else 0
                    pred_count = int(pred_slice[f"is_top_{k}"].sum()) if len(pred_slice) > 0 else 0
                    tp_count = int(pred_slice[f"correct_at_{k}"].sum()) if len(pred_slice) > 0 else 0

                    precision = tp_count / pred_count if pred_count > 0 else np.nan
                    recall = hit_count / len(gt_slice) if len(gt_slice) > 0 else np.nan
                    if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = np.nan

                    base_row[f"hit_at_{k}"] = hit_count
                    base_row[f"pred_count_at_{k}"] = pred_count
                    base_row[f"tp_at_{k}"] = tp_count
                    base_row[f"precision_at_{k}"] = precision
                    base_row[f"recall_at_{k}"] = recall
                    base_row[f"f1_at_{k}"] = f1

                train_count = base_row["train_count"]
                precision_100 = base_row.get("precision_at_100", np.nan)
                recall_100 = base_row.get("recall_at_100", np.nan)
                base_row["precision_div_log_train_count"] = precision_100 / np.log1p(train_count) if train_count > 0 and pd.notna(precision_100) else np.nan
                base_row["recall_div_log_train_count"] = recall_100 / np.log1p(train_count) if train_count > 0 and pd.notna(recall_100) else np.nan
                base_row["is_zero_train_predicate"] = train_count == 0
                rows.append(base_row)

            result = pd.DataFrame(rows).sort_values(["precision_at_100", "recall_at_100"], ascending=[False, False])
            return result.reset_index(drop=True)


        if context is not None:
            predicate_metrics_df = compute_predicate_metrics(
                gt_df=context["tables"]["gt_df"],
                pred_df=context["tables"]["pred_df"],
                label_mapping=context["label_mapping"],
                topk_list=context["topk_list"],
                train_statistics=context["train_statistics"],
                result_dict=context["artifacts"]["result_dict"],
            )
            context["tables"]["predicate_metrics_df"] = predicate_metrics_df

            show_message("### 谓词级表现总表（按 Top-100 精确率和召回率排序）")
            display(predicate_metrics_df.head(30))
        '''
    ),
    markdown_cell(
        """
        ## 8. 三元组级统计分析

        这一节会从更细粒度分析固定谓词内部的不同三元组：

        - 哪些 `(subject, predicate, object)` 组合支持数高但精度仍然低
        - 哪些组合虽然训练样本少，但模型反而更稳定
        - 哪些三元组属于零样本或稀有模式
        """
    ),
    code_cell(
        '''
        def compute_triplet_metrics(gt_df: pd.DataFrame, pred_df: pd.DataFrame, train_statistics: dict) -> pd.DataFrame:
            """按照 triplet_key 聚合测试 GT、预测 TP 和训练频次。"""
            gt_group = (
                gt_df.groupby(
                    ["triplet_key", "triplet_str", "subject_name", "predicate_name", "object_name"],
                    as_index=False,
                )
                .agg(
                    gt_count=("gt_index", "size"),
                    hit_at_20=("matched_at_20", "sum"),
                    hit_at_50=("matched_at_50", "sum"),
                    hit_at_100=("matched_at_100", "sum"),
                )
            )

            pred_group = (
                pred_df.groupby(
                    ["triplet_key", "triplet_str", "subject_name", "predicate_name", "object_name"],
                    as_index=False,
                )
                .agg(
                    pred_count_at_20=("is_top_20", "sum"),
                    pred_count_at_50=("is_top_50", "sum"),
                    pred_count_at_100=("is_top_100", "sum"),
                    tp_at_20=("correct_at_20", "sum"),
                    tp_at_50=("correct_at_50", "sum"),
                    tp_at_100=("correct_at_100", "sum"),
                    avg_triplet_score=("triplet_score", "mean"),
                )
            )

            result = gt_group.merge(
                pred_group,
                on=["triplet_key", "triplet_str", "subject_name", "predicate_name", "object_name"],
                how="outer",
            ).fillna({
                "gt_count": 0,
                "hit_at_20": 0,
                "hit_at_50": 0,
                "hit_at_100": 0,
                "pred_count_at_20": 0,
                "pred_count_at_50": 0,
                "pred_count_at_100": 0,
                "tp_at_20": 0,
                "tp_at_50": 0,
                "tp_at_100": 0,
            })

            for k in [20, 50, 100]:
                result[f"precision_at_{k}"] = np.divide(
                    result[f"tp_at_{k}"],
                    result[f"pred_count_at_{k}"],
                    out=np.full(len(result), np.nan, dtype=np.float64),
                    where=result[f"pred_count_at_{k}"] > 0,
                )
                result[f"recall_at_{k}"] = np.divide(
                    result[f"hit_at_{k}"],
                    result["gt_count"],
                    out=np.full(len(result), np.nan, dtype=np.float64),
                    where=result["gt_count"] > 0,
                )

            result["train_count"] = result["triplet_key"].apply(lambda key: int(train_statistics["triplet_counter"].get(tuple(key), 0)))
            result["is_zero_shot_triplet"] = result["triplet_key"].apply(lambda key: tuple(key) in context["zeroshot_triplets"] if context is not None else False)
            result["is_zero_train_triplet"] = result["train_count"] == 0

            return result.sort_values(["precision_at_100", "recall_at_100", "gt_count"], ascending=[False, False, False]).reset_index(drop=True)


        if context is not None:
            triplet_metrics_df = compute_triplet_metrics(
                gt_df=context["tables"]["gt_df"],
                pred_df=context["tables"]["pred_df"],
                train_statistics=context["train_statistics"],
            )
            context["tables"]["triplet_metrics_df"] = triplet_metrics_df

            show_message("### 三元组级表现总表")
            display(triplet_metrics_df.head(30))
        '''
    ),
    markdown_cell(
        """
        ## 9. 训练频次与性能相关性分析

        这一节重点回答你的核心问题之一：

        > 训练数据越多，模型精确度是否越高？

        我们会同时提供两种视角：

        - 原始散点图：`train_count` 对 `precision_at_100`
        - 归一化比值：`precision_at_100 / log(train_count + 1)`

        这样既能保留直观趋势，也能避免高频谓词单纯因为计数大而主导观察结果。
        """
    ),
    code_cell(
        '''
        def compute_rank_correlation(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
            """输出 Pearson 与 Spearman 两种相关系数，便于快速判断频次和精度的关系。"""
            valid_df = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_df) < 2:
                return pd.DataFrame([{"method": "pearson", "corr": np.nan}, {"method": "spearman", "corr": np.nan}])

            pearson_corr = valid_df[x_col].corr(valid_df[y_col], method="pearson")
            spearman_corr = valid_df[x_col].corr(valid_df[y_col], method="spearman")
            return pd.DataFrame([
                {"method": "pearson", "corr": pearson_corr},
                {"method": "spearman", "corr": spearman_corr},
            ])


        if context is not None:
            predicate_metrics_df = context["tables"]["predicate_metrics_df"].copy()
            predicate_metrics_df["log_train_count"] = np.log1p(predicate_metrics_df["train_count"])
            predicate_metrics_df["train_count_bucket"] = pd.cut(
                predicate_metrics_df["train_count"],
                bins=[-0.1, 0, 10, 100, 1000, np.inf],
                labels=["0", "1-10", "11-100", "101-1000", ">1000"],
            )

            context["tables"]["predicate_train_freq_correlation_df"] = predicate_metrics_df

            show_message("### 相关系数")
            display(compute_rank_correlation(predicate_metrics_df, "train_count", "precision_at_100"))

            if plt is not None:
                fig, axes = plt.subplots(1, 2, figsize=(18, 7))
                axes[0].scatter(predicate_metrics_df["train_count"], predicate_metrics_df["precision_at_100"], alpha=0.8)
                axes[0].set_title("训练频次 vs 谓词精确率（Top-100）")
                axes[0].set_xlabel("train_count")
                axes[0].set_ylabel("precision_at_100")

                axes[1].scatter(predicate_metrics_df["train_count"], predicate_metrics_df["precision_div_log_train_count"], alpha=0.8, color="tab:orange")
                axes[1].set_title("训练频次 vs 频次归一化精确率比值")
                axes[1].set_xlabel("train_count")
                axes[1].set_ylabel("precision / log(train_count + 1)")

                plt.tight_layout()
                plt.show()
            else:
                print("当前环境缺少 matplotlib，已跳过散点图绘制。")

            bucket_summary_df = (
                predicate_metrics_df.groupby("train_count_bucket", dropna=False, as_index=False)
                .agg(
                    predicate_count=("predicate_id", "size"),
                    avg_precision_at_100=("precision_at_100", "mean"),
                    avg_recall_at_100=("recall_at_100", "mean"),
                    avg_ratio=("precision_div_log_train_count", "mean"),
                )
            )
            context["tables"]["train_bucket_summary_df"] = bucket_summary_df

            show_message("### 训练频次分桶汇总")
            display(bucket_summary_df)
        '''
    ),
    markdown_cell(
        """
        ## 10. 长尾 / 零样本 / 谓词混淆切片分析

        这一节补充一些非常必要的切片：

        - 长尾谓词是否显著更难
        - 零训练频次 / 零样本三元组是否几乎全部失效
        - 某个 GT 谓词最容易被混淆成哪些其他谓词
        """
    ),
    code_cell(
        '''
        if context is not None:
            predicate_metrics_df = context["tables"]["predicate_metrics_df"].copy()
            triplet_metrics_df = context["tables"]["triplet_metrics_df"].copy()
            confusion_df = context["tables"]["confusion_df"].copy()

            show_message("### 长尾谓词切片")
            tail_view = predicate_metrics_df.sort_values("train_count", ascending=True).head(20)[[
                "predicate_name", "train_count", "gt_count", "precision_at_100", "recall_at_100", "precision_div_log_train_count"
            ]]
            display(tail_view)

            show_message("### 零训练频次三元组")
            zero_train_triplet_df = triplet_metrics_df.loc[triplet_metrics_df["is_zero_train_triplet"]].sort_values(
                ["gt_count", "precision_at_100"], ascending=[False, False]
            )
            display(zero_train_triplet_df.head(20))

            show_message("### 官方零样本三元组")
            zero_shot_triplet_df = triplet_metrics_df.loc[triplet_metrics_df["is_zero_shot_triplet"]].sort_values(
                ["gt_count", "precision_at_100"], ascending=[False, False]
            )
            display(zero_shot_triplet_df.head(20))

            show_message("### 谓词混淆 Top 表（只统计 Top-100 未命中的 GT）")
            confusion_top_df = (
                confusion_df.loc[~confusion_df["matched_at_100"]]
                .groupby(["gt_predicate_name", "candidate_predicate_name"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
                .sort_values("count", ascending=False)
            )
            context["tables"]["predicate_confusion_top_df"] = confusion_top_df
            display(confusion_top_df.head(30))
        '''
    ),
    markdown_cell(
        """
        ## 11. 代表性案例可视化

        这一节提供一个简单但实用的案例查看器：

        - 支持查看某个谓词下的高置信伪阳性
        - 支持查看某个谓词下的典型漏检
        - 支持查看某个谓词下的正确预测

        如果 `visual_info.json` 或图片文件缺失，函数会自动降级为只显示表格信息。
        """
    ),
    code_cell(
        '''
        def show_image_if_possible(img_file: Optional[str], title: str = "") -> None:
            """尽量显示图片；如果文件不存在，则打印说明。"""
            if not ENABLE_IMAGE_VIS:
                print("图片显示已关闭。")
                return
            if plt is None:
                print("当前环境缺少 matplotlib，无法直接显示图片。")
                return
            if not img_file:
                print("当前样本没有可用的图片路径。")
                return
            img_path = Path(img_file)
            if not img_path.exists():
                print(f"图片不存在: {img_path}")
                return
            image = Image.open(img_path).convert("RGB")
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.axis("off")
            plt.title(title)
            plt.show()


        def show_predicate_cases(predicate_name: str, case_type: str = "false_positive", topk: int = 100, num_cases: int = 3) -> pd.DataFrame:
            """
            按谓词查看代表性样本。

            case_type 支持：
            - false_positive
            - missed_gt
            - correct_prediction
            """
            if context is None:
                print("请先完成前面的数据加载与分析。")
                return pd.DataFrame()

            pred_df = context["tables"]["pred_df"]
            gt_df = context["tables"]["gt_df"]

            if case_type == "false_positive":
                case_df = pred_df.loc[
                    (pred_df["predicate_name"] == predicate_name)
                    & pred_df[f"is_top_{topk}"]
                    & ~pred_df[f"correct_at_{topk}"]
                ].sort_values("triplet_score", ascending=False)
            elif case_type == "missed_gt":
                case_df = gt_df.loc[
                    (gt_df["predicate_name"] == predicate_name)
                    & ~gt_df[f"matched_at_{topk}"]
                ].copy()
            elif case_type == "correct_prediction":
                case_df = pred_df.loc[
                    (pred_df["predicate_name"] == predicate_name)
                    & pred_df[f"correct_at_{topk}"]
                ].sort_values("triplet_score", ascending=False)
            else:
                raise ValueError("case_type 仅支持 false_positive / missed_gt / correct_prediction")

            preview_df = case_df.head(num_cases)
            display(preview_df)

            for _, row in preview_df.iterrows():
                title = f"{case_type} | {row.get('triplet_str', 'N/A')} | image_id={row['image_id']}"
                show_image_if_possible(row.get("img_file"), title=title)

            return preview_df


        if context is not None and len(context["tables"]["predicate_metrics_df"]) > 0:
            error_predicates = (
                context["tables"]["predicate_metrics_df"]
                .sort_values(["precision_at_100", "gt_count"], ascending=[True, False])
                .head(5)["predicate_name"]
                .tolist()
            )
            show_message("### 建议优先检查的低精度谓词")
            display(pd.DataFrame({"predicate_name": error_predicates}))
            print("示例：show_predicate_cases(error_predicates[0], case_type='false_positive', topk=100, num_cases=SHOW_CASES_PER_PREDICATE)")
        '''
    ),
    markdown_cell(
        """
        ## 12. 结果导出

        最后一节把最重要的表导出为 CSV，方便你做后续筛选、画图或写报告。

        默认导出：

        - `predicate_metrics.csv`
        - `triplet_metrics.csv`
        - `missed_gt_triplets.csv`
        - `false_positive_triplets.csv`
        - `predicate_train_freq_correlation.csv`
        """
    ),
    code_cell(
        '''
        if context is not None:
            export_dir = context["export_dir"]
            export_dir.mkdir(parents=True, exist_ok=True)

            export_map = {
                "predicate_metrics.csv": context["tables"].get("predicate_metrics_df"),
                "triplet_metrics.csv": context["tables"].get("triplet_metrics_df"),
                "missed_gt_triplets.csv": context["tables"].get("missed_gt_top100_df"),
                "false_positive_triplets.csv": context["tables"].get("false_positive_top100_df"),
                "predicate_train_freq_correlation.csv": context["tables"].get("predicate_train_freq_correlation_df"),
            }

            exported_files = []
            for file_name, df in export_map.items():
                if df is None or len(df) == 0:
                    continue
                out_path = export_dir / file_name
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                exported_files.append(str(out_path))

            show_message("### 导出完成")
            display(pd.DataFrame({"exported_file": exported_files}))
        '''
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output_path = Path(__file__).resolve().parent / "vg_eval_analysis.ipynb"
output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Notebook 已生成: {output_path}")
