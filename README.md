# Cross-Curriculum Transfer Learning & Multi-Modal Fusion for Student Performance Prediction

> SCI-grade research system based on the OULAD dataset

---

## Overview

This system implements a complete experimental pipeline for a research paper on student performance prediction, featuring:

- **Multi-modal feature fusion**: Sequential behavioral data (LSTM) + tabular statistical/demographic features (XGBoost, CatBoost)
- **Dynamic weight fusion**: Sample-adaptive model ensemble via a learned weight network
- **Early prediction**: 4-week / 8-week / full-course temporal windows
- **Transfer learning**: Leave-One-Module-Out (LOMO) cross-curriculum evaluation
- **Meta-learning**: MAML for few-shot adaptation to new modules
- **Interpretability**: SHAP global + local analysis, modal contribution comparison
- **Statistical tests**: Paired t-test + Wilcoxon signed-rank test with significance labels
- **Auto-generated outputs**: LaTeX tables, publication-ready figures

---

## Project Structure

```
oulad_project/
├── config.py               # All hyperparameters and paths
├── data_preprocessing.py   # Data loading, feature engineering, LOMO/MAML builders
├── models.py               # LSTM, XGBoost, CatBoost, DynamicFusion, MAML
├── evaluation.py           # Metrics, significance tests, SHAP, reporting
├── experiments.py          # Experiment orchestrator (main entry point)
├── utils/
│   └── cross_validation.py # CV utilities and hyperparameter search
├── data/                   # Place OULAD CSV files here
├── outputs/                # CSV results, LaTeX tables
├── figures/                # PNG figures (300 DPI)
├── saved_models/           # Saved model weights
├── logs/                   # Experiment logs
└── requirements.txt
```

---

## Dataset Setup

Download OULAD from: https://analyse.kmi.open.ac.uk/open_dataset

Place these files in `./data/`:

```
studentInfo.csv
studentVle.csv
assessments.csv
studentAssessment.csv
studentRegistration.csv
courses.csv
vle.csv
```

If no data is present, the system automatically generates synthetic data for code verification.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Run all experiments

```bash
cd oulad_project
python experiments.py
```


### PRD-aligned automation scripts

```bash
python run_full.py
python run_lomo.py
python run_ablation.py
python run_lambda_search.py
```

These scripts generate extended evidence files under `outputs/` including: `main_results.csv`, `lomo_results.csv`, `ablation_results.csv`, `lambda_sensitivity.csv`, `weight_distribution.csv`, `diversity_matrix.csv`, `transfer_distance.csv`, and `shap_rank_correlation.csv`.

> Note: this project stores text artifacts only in version control, so loss curves are exported as CSV series in `outputs/loss_curves/*.csv` instead of binary images.

> CUDA acceleration is enabled automatically when available (PyTorch LSTM/HAFM + XGBoost/CatBoost GPU backends with safe CPU fallback).

### Run individual experiments

```python
from data_preprocessing import OULADDatasetBuilder
from experiments import run_standard_comparison, run_lomo_transfer, run_maml
from evaluation import ResultsReporter

reporter = ResultsReporter()
dataset = OULADDatasetBuilder().build()

# Standard comparison
results = run_standard_comparison(dataset, reporter)

# LOMO transfer
lomo = run_lomo_transfer(dataset, reporter)

# MAML meta-learning
maml = run_maml(dataset, reporter)
```

---

## Model Architecture

### LSTM (Sequential Behavioral Data)

```
Input: (N, T=20, D=4)
  → LSTM(hidden=64, layers=2, dropout=0.3)
  → FC(64 → 32) → ReLU → Dropout
  → FC(32 → 1)
Output: (N, 1)
```

### Dynamic Fusion

```
Input: [y_LSTM, y_XGB, y_CatBoost] + tabular features
  Weight network: FC(F→32) → ReLU → FC(32→3) → Softmax
  y_final = w1·y_LSTM + w2·y_XGB + w3·y_CatBoost
```

### MAML

```
Meta-model: MLP(F → 64 → 32 → 1)
Per-module inner loop: SGD (5 steps, lr=0.01)
Outer loop: Adam (lr=0.001)
```

---

## Outputs

After running, outputs will be generated in:

| Path | Content |
|---|---|
| `outputs/standard_comparison_results.csv` | Main model metrics |
| `outputs/main_results.tex` | LaTeX table (ready for paper) |
| `outputs/lomo_results.csv` | LOMO per-module metrics |
| `outputs/maml_results.csv` | MAML metrics |
| `outputs/significance_tests.csv` | p-values and significance labels |
| `outputs/shap_importance.csv` | Feature importance ranking |
| `figures/comparison_RMSE.png` | Model comparison bar chart |
| `figures/early_prediction_RMSE.png` | Early prediction line chart |
| `figures/lomo_transfer_RMSE.png` | Transfer learning comparison |
| `figures/shap_summary.png` | SHAP summary plot |
| `figures/shap_modal_contribution.png` | Modal contribution bar chart |

---

## Key Features (Innovation Points)

1. **Multi-modal fusion with dynamic weights** — learns to weight models per-sample based on student context
2. **Strict leakage prevention** — grades reconstructed from weighted assessments, not raw `final_result`
3. **LOMO transfer validation** — 7-module cross-curriculum generalization
4. **MAML few-shot learning** — fast adaptation to new course modules
5. **Early prediction** — performance under 4/8-week information constraints
6. **SHAP interpretability** — behavioral vs demographic modal contribution quantification
7. **Statistical rigor** — paired t-test + Wilcoxon for all comparisons

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2025,
  title={Cross-Curriculum Transfer Learning with Multi-Modal Fusion for Student Performance Prediction},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

---

## License

For academic research use only.

---

## MT-HAFNet Modular Scaffold (New)

A PRD-aligned modular implementation is also provided:

```
preprocess/data_builder.py
models/lstm.py
models/xgb.py
models/cat.py
models/hafm.py
models/baselines.py
models/__init__.py   # includes legacy import compatibility
loss/unified_loss.py
train/train_full.py
train/train_lomo.py
train/train_baselines.py
evaluation/metrics.py
evaluation/statistics.py
evaluation/__init__.py  # includes legacy import compatibility
explain/shap_analysis.py
run_all.py
```

Smoke run:

```bash
python run_all.py
```


## Junyi 精细化行为权重分析模块（新增）

为避免影响原有 OULAD 训练流程，新增了独立的 Junyi 模块：

- `junyi/dataloader.py`：
  - 按 `user_id + timestamp` 构建行为序列
  - `MAX_SEQ_LEN` 截断/补齐
  - 多模态输入：`exercise`、`correct`、`elapsed_time/hint_used`
  - 可选根据 `junyi_Exercise_table.csv` 构建先修关系邻接矩阵
- `src/models/dynamic_junyi.py`：
  - 动态权重生成器（时间注意力）
  - 可选图融合门控单元（行为 vs 知识图）
  - 统一优化目标：`BCE + λ * attention_regularization`
- `experiments/exp_generalization.py`：按习题ID 80/20 未见迁移划分，输出 Accuracy/AUC
- `experiments/exp_robustness.py`：测试集噪声注入（标签翻转/耗时置零），输出性能衰减
- `experiments/vis_weights.py`：导出注意力权重并绘制热力图

示例：

```bash
python experiments/exp_generalization.py --log_csv data/junyi_ProblemLog_original.csv --exercise_csv data/junyi_Exercise_table.csv
python experiments/exp_robustness.py --log_csv data/junyi_ProblemLog_original.csv
python experiments/vis_weights.py --log_csv data/junyi_ProblemLog_original.csv --sample_idx 0
```
