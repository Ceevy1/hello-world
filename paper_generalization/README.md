# 小样本跨域泛化实验（重构版）

本目录与根目录脚本提供一个**从零实现**的小样本（<200）实验流水线，目标是复现以下输出：
- 数据诊断报告
- 多基线交叉验证结果
- 轻量 DynamicFusion 预训练 + 冻结微调结果
- 数据增强产物
- 蒸馏学生模型结果
- 评估指标（RMSE/MAE/R² + 95%CI）
- 统计显著性检验（p-value + Cohen's d）
- SHAP 解释图与特征重要性

## 推荐执行顺序

```bash
python data_diagnosis.py --input data/student_scores.csv --output outputs/diag_report.txt
python train_baselines.py --data data/student_scores.csv --cv 5 --out outputs/baseline_results.csv

python train_dynamicfusion.py --domain OULAD --source_csv data/student_scores.csv --epochs 100 --save outputs/model_oulad.pth
python fine_tune.py --model outputs/model_oulad.pth --data data/student_scores.csv --freeze_encoder True --epochs 80 --out outputs/model_ft.pth

python augment_data.py --input data/student_scores.csv --method mixup --target outputs/augmented_data.csv
python distill_student.py --teacher outputs/model_oulad.pth --data data/student_scores.csv --out outputs/student_model.pth

python evaluate_model.py --model outputs/model_ft.pth --data data/student_scores.csv --metrics RMSE MAE R2 --out outputs/eval_results.csv
python pg_shap_analysis.py --model outputs/student_model.pth --data data/student_scores.csv --out outputs/shap_plots

python stats_test.py --baseline outputs/baseline_results_rf_folds.csv --candidate outputs/model_ft_cv_metrics.csv --out outputs/stats_report.csv
```

## 文件说明
- `paper_generalization/common.py`: 数据读取、模态分组、指标和CI工具。
- `paper_generalization/models.py`: 轻量 DynamicFusion 与蒸馏学生网络。
- 根目录脚本：对应论文实验环节，可独立执行。

> 注意：当前 `data/student_scores.csv` 仅 5 条样本，统计稳定性有限。建议实际实验替换为完整自建数据并保持同列结构。
