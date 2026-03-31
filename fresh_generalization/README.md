# DynamicFusion 泛化能力实验框架（重构版）

该目录提供一个**从零实现**的跨域泛化实验脚本，不依赖仓库中原有实验代码。

## 设计目标
- 支持将源域（OULAD）迁移到目标域（`student_scores`）进行性能评估。
- 包含：语义特征对齐、共享编码器预训练、DynamicFusion 训练、目标域微调、基线对比、分布差异评估、反事实干预分析。

## 输入数据
- 目标域：默认读取 `/data/student_scores.csv`，若不存在自动回退到 `data/student_scores.csv`。
- 源域：可通过 `--oulad-data` 传入 CSV。若未提供，自动构建“synthetic proxy source”（用于流程验证与可运行性测试）。

## 运行
```bash
python fresh_generalization/run_dynamicfusion_generalization.py \
  --self-data /data/student_scores.csv \
  --output-dir fresh_generalization/outputs
```

若你已准备 OULAD 对齐前数据（如 studentInfo / 聚合特征表），可以：
```bash
python fresh_generalization/run_dynamicfusion_generalization.py \
  --self-data /data/student_scores.csv \
  --oulad-data /path/to/oulad.csv \
  --output-dir fresh_generalization/outputs_oulad
```

## 输出文件
- `metrics_summary.json`: 主指标汇总（zero-shot、fine-tune、基线、泛化下降率、注意力均值、干预效果、CI）。
- `performance_table.csv`: 各模型 RMSE/MAE/R² 对比。
- `distribution_shift.csv`: 各特征 Wasserstein 距离。

## 注意事项
- 当前 `student_scores.csv` 样本极少时，脚本会自动执行轻量数据增强以保证神经网络可训练。
- 若要用于正式论文结果，请替换为真实 OULAD 源域数据，并禁用/调整增强策略。
