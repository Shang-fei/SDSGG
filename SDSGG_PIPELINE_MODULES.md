# SDSGG Pipeline 模块说明

## 总体流程

SDSGG 主流程可以概括为：
数据与配置加载 -> 模型构建 -> 关系对采样 -> 关系特征提取 -> 关系预测与损失计算 -> 训练/验证/测试 -> 指标评估与结果输出。

## 1. 入口脚本层（tools）

### tools/relation_train_net.py
- 关系模型主训练入口。
- 读取配置、构建模型、加载预训练权重、组织 train/val dataloader。
- 执行训练循环（前向、loss、反向、优化、保存 checkpoint、周期验证）。

### tools/relation_test_net.py
- 关系模型主测试入口。
- 加载权重并创建 test dataloader。
- 调用 inference 流程，输出关系预测结果并触发评估。

### tools/detector_pretrain_net.py
- 目标检测分支预训练入口。
- 用于先训检测器，为 relation head 提供更稳的目标框与特征。

### tools/detector_pretest_net.py
- 检测分支测试入口。

### tools/relation_train_net_svm.py
- 关系训练实验脚本（研究/分析用途，非主线训练入口）。

### script.txt
- 提供 base/novel/semantic/total 设置下的训练与测试命令模板。

## 2. 配置层（configs）

- 通过 yaml 配置模型结构、数据集、优化器、评估协议、batch size 等。
- OV_SETTING 的 TRAIN_PART/VAL_PART/TEST_PART 决定评测子集：base、novel、semantic、total。

## 3. 数据层（maskrcnn_benchmark/data）

- 负责数据集读取与 dataloader 构建。
- 核心结构：
1. ImageList：支持不同分辨率图像的 batch 封装。
2. BoxList：统一封装 bbox 及 labels/relation 等字段。

## 4. 模型装配层（maskrcnn_benchmark/modeling）

- build_detection_model 统一组装 backbone、rpn、roi_heads。
- 检测头负责对象定位与分类。
- 关系头负责 subject-object 谓词预测。

## 5. 关系头核心模块（SDSGG 关键）

目录：maskrcnn_benchmark/modeling/roi_heads/relation_head

### relation_head.py
- 关系分支总控模块。
- 串联采样、特征提取、预测器、损失和推理后处理。

### sampling.py
- 构造关系对（subject-object pair）。
- 训练时进行正负关系采样；测试时生成候选 pair。

### roi_relation_feature_extractors.py
- 提取关系特征。
- 对 union box 做视觉特征池化，并融合 head/tail 空间布局信息。

### roi_relation_predictors.py
- 关系预测核心。
- 结合上下文编码器与 CLIP 语义空间，输出关系 logits。
- SDSGG 的 scene-specific 描述与适配机制主要在该模块体现。

### model_motifs.py / model_vctree.py / model_transformer.py / model_vtranse.py
- 不同关系上下文建模器，实现不同的上下文编码策略。

### loss.py
- 关系训练损失计算。
- 除标准分类损失外，融合 description_relation_loss.csv 等先验信息。

### inference.py
- 关系推理后处理。
- 将 logits 转为可评估的关系 triplet 输出。

## 6. SDSGG 特有的 Scene-specific 描述模块

### CSV 先验文件
- 关键文件：description_relation.csv、description_relation_loss.csv、filter_total.csv 等。
- 作用：提供关系-对象层面的场景描述先验，辅助关系分类器动态调整。

### 生成说明
- 文档：CSV_GENERATION_README.md
- 定义了描述共现/矛盾等类别及分数生成方式，用于构建上述 CSV。

## 7. 训练与推理公共支撑

### engine
- trainer/inference：统一训练与推理流程组织。

### solver
- 优化器与学习率调度策略。

### utils/checkpoint
- 模型加载、断点续训、周期保存。

### utils/comm 与 logger
- 分布式同步、日志记录、指标聚合。

## 8. 评估模块

- 指标定义见 METRICS.md。
- 常见指标：R@K、ng-R@K、mR@K、ng-mR@K、zR@K、ng-zR@K。
- 推理结果通常写入 OUTPUT_DIR/inference/<dataset_name>/，并据此计算指标。

## 9. 典型执行链路

### 训练
1. 运行 relation_train_net.py + config。
2. 加载预训练检测器权重。
3. dataloader 提供图像与标注。
4. relation head 执行采样 -> 特征 -> 预测 -> loss。
5. 反向传播与更新。
6. 周期验证并保存模型。

### 测试
1. 运行 relation_test_net.py + config。
2. 加载训练好的权重。
3. 生成候选关系对并推理。
4. 后处理输出 triplets。
5. 计算并汇报 SGG 指标。
