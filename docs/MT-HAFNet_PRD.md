# MT-HAFNet：基于统一优化与跨模块迁移的学生成绩预测框架（工程PRD）

本文件将需求映射到代码目录，支持后续开发、复现与投稿实验组织。

## 代码落地目录

- `preprocess/data_builder.py`: 标签重构、序列/表格特征、LOMO/LOPO/随机划分、早期截断。
- `models/lstm.py`: LSTM(hidden=128)+Dropout(0.3)+FC。
- `models/xgb.py`: XGBoost 回归模型。
- `models/cat.py`: CatBoost 回归模型。
- `models/hafm.py`: 动态融合权重网络（F→64→3 + Softmax）。
- `loss/unified_loss.py`: 统一损失（reg + transfer + diversity + stability）与 λ 网格定义。
- `train/train_full.py`: 全流程训练与融合评估。
- `train/train_lomo.py`: Leave-One-Module-Out 迁移评估。
- `evaluation/metrics.py`: RMSE/MAE/R²/Std/95%CI。
- `evaluation/statistics.py`: Paired t-test / Wilcoxon。
- `explain/shap_analysis.py`: SHAP summary/bar 输出。
- `run_all.py`: 端到端烟囱验证入口（含全量、4周、LOMO、显著性）。

## 说明

当前实现提供可运行骨架与关键公式接口；接入 OULAD 原始表后可直接扩展为完整论文实验脚本。
