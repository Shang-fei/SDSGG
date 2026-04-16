# Configs Summary

这份文档总结 `configs/` 目录中各类配置文件的作用，以及你在复现 SDSGG 时最需要关注的内容：**用什么数据集、用什么 backbone、先训练什么、后训练什么**。

## 1. `configs/` 目录整体在干什么

`configs/` 里混了两类东西：

1. **这个项目真正会用到的 SDSGG / SGG 配置**
   - `configs/e2e_relation_detector_*.yaml`
   - `configs/e2e_relation_*.yaml`

2. **继承自 `maskrcnn-benchmark` 的通用示例配置**
   - `configs/caffe2/`
   - `configs/maskrcnn_benchmark_models/`
   - `configs/quick_schedules/`
   - `configs/retinanet/`
   - `configs/pascal_voc/`
   - `configs/cityscapes/`
   - `configs/dcn/`
   - `configs/gn_baselines/`
   - `configs/test_time_aug/`

如果你的目标是 **把 SDSGG 当 baseline 跑起来**，重点只需要看根目录下这几类：

- `configs/e2e_relation_detector_R_101_FPN_1x.yaml`
- `configs/e2e_relation_detector_VGG16_1x.yaml`
- `configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml`
- `configs/e2e_relation_R_101_FPN_1x.yaml`
- `configs/e2e_relation_VGG16_1x.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_base.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_novel.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_semantic.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_clip_GQA200.yaml`

## 2. 你真正需要理解的两阶段配置

这个仓库的主线是两阶段：

1. **先训练 detector**
   - 配置文件名里通常有 `relation_detector`
   - 特征是 `MODEL.RELATION_ON: False`
   - 用于训练物体检测器，给后续 relation 训练提供 checkpoint

2. **再训练 relation model**
   - 配置文件名里通常是 `e2e_relation_*.yaml`
   - 特征是 `MODEL.RELATION_ON: True`
   - 在 detector checkpoint 基础上训练关系预测头

所以：

- `detector_pretrain_net.py` 对应 `configs/e2e_relation_detector_*.yaml`
- `relation_train_net.py` 对应 `configs/e2e_relation_*.yaml`

## 3. 数据集配置总结

### 3.1 Visual Genome（VG）

绝大多数配置默认都在用 VG。

常见数据集名：

- `VG_stanford_filtered_with_attribute_train`
- `VG_stanford_filtered_with_attribute_val`
- `VG_stanford_filtered_with_attribute_test`

这些名字在 `maskrcnn_benchmark/config/paths_catalog.py` 中映射到实际路径：

- 图片目录：`datasets/vg/VG_100K`
- 标注文件：`datasets/vg/VG-SGG-with-attri.h5`
- 字典文件：`datasets/vg/VG-SGG-dicts-with-attri.json`
- 图像元信息：`datasets/vg/image_data.json`

对应代码位置：

- `maskrcnn_benchmark/config/paths_catalog.py`
- `DATASET.md`

### 3.2 GQA

GQA 只在专门的配置里出现：

- `configs/e2e_relation_X_101_32_8_FPN_1x_clip_GQA200.yaml`

对应数据集名：

- `GQA_200_train`
- `GQA_200_val`
- `GQA_200_test`

映射路径：

- 图片目录：`datasets/gqa/images`
- id / 类别信息：`datasets/gqa/GQA_200_ID_Info.json`
- train：`datasets/gqa/GQA_200_Train.json`
- test：`datasets/gqa/GQA_200_Test.json`

### 3.3 detector 训练和 relation 训练的数据过滤差异

`paths_catalog.py` 里有一个关键逻辑：

- 如果 `MODEL.RELATION_ON=False`，不会强制过滤空关系图像
- 如果 `MODEL.RELATION_ON=True`，会过滤没有 relation 的样本

这意味着：

- **训练 detector** 时，目标是学物体检测，所以会尽量保留更多图像
- **训练 relation** 时，目标是学关系，所以数据集会强调 relation annotations

## 4. backbone 配置总结

根目录真正相关的 backbone 主要有三种：

### 4.1 `R-101-FPN`

出现于：

- `configs/e2e_relation_detector_R_101_FPN_1x.yaml`
- `configs/e2e_relation_R_101_FPN_1x.yaml`

特点：

- ResNet-101 + FPN
- `MODEL.WEIGHT` 用 `ImageNetPretrained/MSRA/R-101`
- 较传统，结构清晰

### 4.2 `VGG-16`

出现于：

- `configs/e2e_relation_detector_VGG16_1x.yaml`
- `configs/e2e_relation_VGG16_1x.yaml`

特点：

- 不用 FPN 的 RPN
- `ANCHOR_STRIDE` 往往是 `(16,)`
- 一般是老 baseline 风格

### 4.3 `X-101-32x8d`

出现于：

- `configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_base.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_novel.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_semantic.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_clip_GQA200.yaml`

特点：

- 实际初始化权重来自 `ImageNetPretrained/FAIR/20171220/X-101-32x8d`
- `NUM_GROUPS: 32`
- `WIDTH_PER_GROUP: 8`
- 是这个项目里最像“主力配置”的 backbone 系列

注意：

- 配置里 `BACKBONE.CONV_BODY` 有时写成 `"R-101-FPN"`，但同时配了 `NUM_GROUPS: 32` 和 `WIDTH_PER_GROUP: 8`，这实际上是在走 ResNeXt-101-32x8d 这条变体
- 所以判断 backbone 不能只看 `CONV_BODY` 一行，还要结合 `MODEL.WEIGHT` 和 `RESNETS.NUM_GROUPS`

## 5. detector 配置文件分别意味着什么

### `configs/e2e_relation_detector_R_101_FPN_1x.yaml`

- 用 VG 数据集
- `MODEL.RELATION_ON: False`
- backbone 是 `R-101-FPN`
- `ROI_BOX_HEAD.NUM_CLASSES: 151`
- 更标准的 VG detector 预训练方案

### `configs/e2e_relation_detector_VGG16_1x.yaml`

- 用 VG 数据集
- `MODEL.RELATION_ON: False`
- backbone 是 `VGG-16`
- 比较老派，不是我最推荐的起点

### `configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml`

- 用 VG 数据集
- `MODEL.RELATION_ON: False`
- backbone 是 ResNeXt-101-32x8d 风格
- `ROI_BOX_HEAD.NUM_CLASSES: 201`

这里有个需要特别注意的点：

- 同目录下大多数 VG relation 配置 `ROI_BOX_HEAD.NUM_CLASSES` 是 `151`
- 但这个 detector 配置里写成了 `201`

这意味着如果你后续打算和某个 relation 配置配套使用，**要确认 detector checkpoint 的类别数和 relation 配置一致**。  
从仓库里其它主关系配置看，VG 主线更常见的是 `151` 类设置，因此这个 `201` 值需要你在实际开训前再核对一次是否是作者有意修改，还是只服务于某个特定实验版本。

## 6. relation 配置文件分别意味着什么

### 6.1 传统 SGG / baseline 系列

#### `configs/e2e_relation_R_101_FPN_1x.yaml`

- VG 数据集
- `R-101-FPN`
- `MODEL.RELATION_ON: True`
- `ROI_RELATION_HEAD.PREDICTOR: "VCTreePredictor"`
- 偏传统 scene graph generation baseline

#### `configs/e2e_relation_VGG16_1x.yaml`

- VG 数据集
- `VGG-16`
- `PREDICTOR: "VCTreePredictor"`
- 更像旧版 baseline

#### `configs/e2e_relation_X_101_32_8_FPN_1x.yaml`

- VG 数据集
- ResNeXt-101-32x8d 风格 backbone
- `PREDICTOR: "CausalAnalysisPredictor"`
- 这是来自上游 Scene Graph Benchmark / unbiased SGG 体系的一条重要 baseline

### 6.2 SDSGG / open-vocabulary 评测系列

这些文件结构很接近，只是关系类别划分不同：

- `configs/e2e_relation_X_101_32_8_FPN_1x_base.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_novel.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_semantic.yaml`
- `configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml`

它们的共同点：

- VG 数据集
- ResNeXt-101-32x8d 风格 backbone
- `PREDICTOR: "CausalAnalysisPredictor"` 作为配置默认值
- 运行命令时经常会通过命令行覆写成 `ClipPredictor`

主要差异：

- `NUM_CLASSES` 不同
- 对应不同 open-vocabulary 关系集合

大致对应关系：

- `base`: 36 类
- `novel`: 16 类附近（以文件实际值为准）
- `semantic`: 25 类
- `total`: 51 类

### 6.3 GQA + Clip predictor

#### `configs/e2e_relation_X_101_32_8_FPN_1x_clip_GQA200.yaml`

- 数据集切到 `GQA_200_*`
- object 类别数变成 `201`
- attribute 数变成 `501`
- relation 类别数变成 `101`
- `ROI_RELATION_HEAD.PREDICTOR: "ClipPredictor"`

这份配置更接近 SDSGG 论文最终那类 “CLIP + scene-specific descriptions” 风格实验。

## 7. 作为 baseline，你最需要关心哪些字段

### 7.1 判断当前配置是 detector 还是 relation

- `MODEL.RELATION_ON: False` → detector 预训练
- `MODEL.RELATION_ON: True` → relation 训练 / 测试

### 7.2 判断当前配置用什么数据集

看 `DATASETS`：

- VG：
  - `VG_stanford_filtered_with_attribute_train`
  - `VG_stanford_filtered_with_attribute_val`
  - `VG_stanford_filtered_with_attribute_test`
- GQA：
  - `GQA_200_train`
  - `GQA_200_val`
  - `GQA_200_test`

### 7.3 判断当前配置用什么 backbone

看这三部分一起判断：

- `MODEL.WEIGHT`
- `MODEL.BACKBONE.CONV_BODY`
- `MODEL.RESNETS.NUM_GROUPS` / `WIDTH_PER_GROUP`

### 7.4 判断当前关系预测器是什么

看：

- `MODEL.ROI_RELATION_HEAD.PREDICTOR`

常见选项：

- `VCTreePredictor`
- `CausalAnalysisPredictor`
- `ClipPredictor`
- `GQAClipPredictor`（一般通过命令行覆写）

## 8. 我对你当前任务的实际建议

你现在没有作者提供的 detector 权重，因此主线应该是：

1. **先决定数据集**
   - 如果你想最快复现主线，优先用 **VG**
   - 如果你目标是论文里 GQA 结果，再走 GQA

2. **先决定 backbone**
   - 最稳妥：`R-101-FPN`
   - 更贴近项目主力配置：`X-101-32x8d`
   - 不推荐一开始用：`VGG-16`

3. **先训练 detector**
   - 对应 `tools/detector_pretrain_net.py`
   - 配 `configs/e2e_relation_detector_*.yaml`

4. **再训练 relation**
   - 对应 `tools/relation_train_net.py`
   - 配 `configs/e2e_relation_*.yaml`
   - 并通过 `MODEL.PRETRAINED_DETECTOR_CKPT` 指向 detector checkpoint

## 9. 推荐起步路线

### 路线 A：更稳妥、便于排错

- 数据集：VG
- detector 配置：`configs/e2e_relation_detector_R_101_FPN_1x.yaml`
- relation 配置：`configs/e2e_relation_R_101_FPN_1x.yaml`

优点：

- 类别数更一致
- 配置更传统
- 更容易先把流程打通

### 路线 B：更接近论文后续主力实验

- 数据集：VG
- detector 配置：`configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml`
- relation 配置：`configs/e2e_relation_X_101_32_8_FPN_1x_total.yaml`

但这条路在开训前请先检查：

- detector 的 `ROI_BOX_HEAD.NUM_CLASSES`
- relation 的 `ROI_BOX_HEAD.NUM_CLASSES`

是否一致；如果不一致，需要先明确是数据设置不同，还是配置遗留问题。

## 10. 你现在至少要准备哪些文件

如果先做 VG：

- `datasets/vg/VG_100K/` 图片
- `datasets/vg/VG-SGG-with-attri.h5`
- `datasets/vg/VG-SGG-dicts-with-attri.json`
- `datasets/vg/image_data.json`

如果做 GQA：

- `datasets/gqa/images/`
- `datasets/gqa/GQA_200_ID_Info.json`
- `datasets/gqa/GQA_200_Train.json`
- `datasets/gqa/GQA_200_Test.json`

## 11. 一句话总结

对你当前最重要的结论是：

- `configs/e2e_relation_detector_*.yaml` 是 **第一阶段 detector 训练**
- `configs/e2e_relation_*.yaml` 是 **第二阶段 relation 训练**
- **VG + R-101-FPN** 是最稳妥起点
- **VG + X-101-32x8d** 更接近项目主力设定，但需要你先核对类别数一致性

