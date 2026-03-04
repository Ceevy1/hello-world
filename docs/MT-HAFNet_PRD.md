# MT-HAFNet：基于统一优化与跨模块迁移的学生成绩预测框架（工程PRD）

本文件将需求映射到代码目录，支持后续开发、复现与投稿实验组织。

## 代码落地目录

- `preprocess/data_builder.py`: 标签重构、序列/表格特征、LOMO/LOPO/随机划分、早期截断。
- `models/lstm.py`: LSTM(hidden=128)+Dropout(0.3)+FC。
- `models/xgb.py`: XGBoost 回归模型。
- `models/cat.py`: CatBoost 回归模型。
- `models/hafm.py`: 动态融合权重网络（F→64→3 + Softmax）。
- `models/baselines.py`: 论文对比所需基模型集合（10个）：Linear/Ridge/Lasso/ElasticNet/SVR/KNN/RF/ExtraTrees/GBDT/AdaBoost。
- `models/__init__.py`: 统一导出入口，补全并兼容旧实验脚本引用（`from models import ...`）。
- `loss/unified_loss.py`: 统一损失（reg + transfer + diversity + stability）与 λ 网格定义。
- `train/train_full.py`: 全流程训练与融合评估（含 HAFM 参数训练）。
- `train/train_lomo.py`: Leave-One-Module-Out 迁移评估。
- `train/train_baselines.py`: 全部基模型训练与评估。
- `evaluation/metrics.py`: RMSE/MAE/R²/Std/95%CI。
- `evaluation/statistics.py`: Paired t-test / Wilcoxon。
- `evaluation/__init__.py`: 评估模块统一出口，兼容旧实验引用。
- `explain/shap_analysis.py`: SHAP summary/bar 输出。
- `run_all.py`: 端到端验证入口（基模型、全量、4周、LOMO、显著性）。

## 说明

当前实现已补齐模型目录基础模型与缺失引用桥接，可直接在现有 `main.py/experiments.py` 上继续扩展完整论文实验。
