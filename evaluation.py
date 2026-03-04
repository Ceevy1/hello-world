"""
Evaluation Module
Metrics, statistical significance tests, SHAP analysis, and result reporting.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from config import FIGURE_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


# ============================================================
# 1. Regression Metrics
# ============================================================
def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute RMSE, MAE, R²."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute accuracy, macro-F1 for classification."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    y_true = np.array(y_true).flatten().astype(int)
    y_pred = np.array(y_pred).flatten().astype(int)
    acc  = float(accuracy_score(y_true, y_pred))
    f1   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"Accuracy": acc, "F1_macro": f1}


# ============================================================
# 2. Statistical Significance Tests
# ============================================================
class SignificanceTester:
    """
    Paired t-test and Wilcoxon signed-rank test
    for comparing model performance across folds.
    """

    @staticmethod
    def paired_ttest(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict:
        """
        Compare absolute errors between model A and model B.
        H0: mean(|e_a|) == mean(|e_b|)
        """
        errors_a = np.array(errors_a).flatten()
        errors_b = np.array(errors_b).flatten()
        min_len = min(len(errors_a), len(errors_b))
        t_stat, p_val = stats.ttest_rel(errors_a[:min_len], errors_b[:min_len])
        return {
            "test": "paired_t-test",
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
            "significance_label": SignificanceTester._sig_label(p_val),
        }

    @staticmethod
    def wilcoxon_test(errors_a: np.ndarray, errors_b: np.ndarray) -> Dict:
        """Wilcoxon signed-rank test (non-parametric)."""
        errors_a = np.array(errors_a).flatten()
        errors_b = np.array(errors_b).flatten()
        min_len = min(len(errors_a), len(errors_b))
        try:
            stat, p_val = stats.wilcoxon(errors_a[:min_len], errors_b[:min_len])
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {"test": "wilcoxon", "p_value": 1.0, "significant": False}
        return {
            "test": "wilcoxon",
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant": p_val < 0.05,
            "significance_label": SignificanceTester._sig_label(p_val),
        }

    @staticmethod
    def _sig_label(p_val: float) -> str:
        if p_val < 0.001: return "***"
        if p_val < 0.01:  return "**"
        if p_val < 0.05:  return "*"
        return "ns"

    def compare_all(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        baseline: str = "LSTM",
    ) -> pd.DataFrame:
        """Compare all models against baseline using both tests."""
        results = []
        baseline_errors = np.abs(y_true - predictions[baseline])

        for model_name, preds in predictions.items():
            if model_name == baseline:
                continue
            model_errors = np.abs(y_true - preds)
            t_res = self.paired_ttest(baseline_errors, model_errors)
            w_res = self.wilcoxon_test(baseline_errors, model_errors)
            results.append({
                "comparison": f"{baseline} vs {model_name}",
                "t_pvalue": t_res["p_value"],
                "t_significant": t_res["significant"],
                "t_label": t_res["significance_label"],
                "w_pvalue": w_res["p_value"],
                "w_significant": w_res["significant"],
                "w_label": w_res.get("significance_label", ""),
            })
        return pd.DataFrame(results)


# ============================================================
# 3. SHAP Analysis
# ============================================================
class SHAPAnalyzer:
    """
    SHAP-based interpretability analysis.
    Supports XGBoost/CatBoost tree explainers and kernel explainer fallback.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None

    def explain_tree(self, model, X: np.ndarray, sample_size: int = 500):
        """Tree SHAP for XGBoost/CatBoost/RF."""
        try:
            import shap
            # Sample for efficiency
            if len(X) > sample_size:
                idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[idx]
            else:
                X_sample = X

            self.explainer = shap.TreeExplainer(model)
            self.shap_values = self.explainer.shap_values(X_sample)
            self.X_sample = X_sample
            logger.info(f"SHAP values computed: shape={np.array(self.shap_values).shape}")
            return self.shap_values
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer failed: {e}. Trying KernelExplainer.")
            return self.explain_kernel(model.predict, X, sample_size=100)

    def explain_kernel(self, predict_fn, X: np.ndarray, sample_size: int = 100):
        """Kernel SHAP for black-box models."""
        try:
            import shap
            background = shap.sample(X, min(50, len(X)))
            if len(X) > sample_size:
                idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[idx]
            else:
                X_sample = X
            self.explainer = shap.KernelExplainer(predict_fn, background)
            self.shap_values = self.explainer.shap_values(X_sample, nsamples=50)
            self.X_sample = X_sample
            return self.shap_values
        except Exception as e:
            logger.error(f"SHAP KernelExplainer failed: {e}")
            return None

    def get_global_importance(self) -> pd.DataFrame:
        """Return mean absolute SHAP values per feature."""
        if self.shap_values is None:
            return pd.DataFrame()
        vals = np.array(self.shap_values)
        if vals.ndim == 3:
            vals = vals.mean(axis=0)
        mean_abs = np.abs(vals).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs)],
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
        return df

    def plot_summary(self, save_path: str = None):
        """Generate SHAP summary plot."""
        try:
            import shap
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                self.shap_values,
                self.X_sample,
                feature_names=self.feature_names[:self.X_sample.shape[1]],
                show=False,
            )
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"SHAP summary plot saved: {save_path}")
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")
            self._plot_fallback(save_path)

    def _plot_fallback(self, save_path: str = None):
        """Fallback bar chart of feature importance."""
        import matplotlib.pyplot as plt
        df = self.get_global_importance()
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        top_df = df.head(15)
        ax.barh(top_df["feature"][::-1], top_df["mean_abs_shap"][::-1], color="#4472C4")
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title("Feature Importance (SHAP)", fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# ============================================================
# 4. Results Reporter
# ============================================================
class ResultsReporter:
    """
    Auto-generate comparison tables, figures, and LaTeX output.
    """

    def __init__(self, output_dir: str = OUTPUT_DIR, figure_dir: str = FIGURE_DIR):
        self.output_dir = output_dir
        self.figure_dir = figure_dir

    def compile_metrics(
        self,
        results: Dict[str, Dict],
        experiment_name: str = "main",
    ) -> pd.DataFrame:
        """
        results: {model_name: {RMSE, MAE, R2, ...}}
        """
        rows = []
        for model, metrics in results.items():
            row = {"Model": model}
            row.update(metrics)
            rows.append(row)
        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, f"{experiment_name}_results.csv")
        df.to_csv(path, index=False)
        logger.info(f"Results saved: {path}")
        return df

    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = "RMSE",
        title: str = "Model Comparison",
        save_name: str = "model_comparison.png",
    ):
        """Bar chart comparing models on a given metric."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0", "#00B0F0", "#92D050"]
        models = results_df["Model"].tolist()
        values = results_df[metric].tolist()

        bars = ax.bar(models, values, color=colors[:len(models)], width=0.6, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Model", fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        plt.tight_layout()
        save_path = os.path.join(self.figure_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Comparison plot saved: {save_path}")
        return save_path

    def plot_early_prediction_curve(
        self,
        window_results: Dict[str, Dict],
        metric: str = "RMSE",
        save_name: str = "early_prediction_curve.png",
    ):
        """
        Line chart showing performance vs prediction window.
        window_results: {"week4": {...}, "week8": {...}, "full": {...}}
        """
        import matplotlib.pyplot as plt

        windows = list(window_results.keys())
        models = set()
        for w_res in window_results.values():
            models.update(w_res.keys())
        models = list(models)

        fig, ax = plt.subplots(figsize=(9, 6))
        markers = ["o", "s", "^", "D", "v", "<", ">"]
        colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0"]

        for i, model in enumerate(models):
            y_vals = []
            x_labels = []
            for w in windows:
                if model in window_results[w]:
                    y_vals.append(window_results[w][model].get(metric, np.nan))
                    x_labels.append(w)
            ax.plot(x_labels, y_vals,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linewidth=2, markersize=8, label=model)

        ax.set_xlabel("Prediction Window", fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f"Early Prediction Performance ({metric})", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(self.figure_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Early prediction curve saved: {save_path}")

    def plot_transfer_comparison(
        self,
        lomo_results: Dict[str, Dict],
        metric: str = "RMSE",
        save_name: str = "transfer_comparison.png",
    ):
        """
        Bar chart per module for LOMO transfer results.
        lomo_results: {module: {model_name: metric_value}}
        """
        import matplotlib.pyplot as plt

        modules = list(lomo_results.keys())
        if not modules:
            return

        models = list(lomo_results[modules[0]].keys())
        x = np.arange(len(modules))
        width = 0.8 / len(models)
        colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0"]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, model in enumerate(models):
            vals = [lomo_results[m].get(model, {}).get(metric, 0) for m in modules]
            ax.bar(x + i * width, vals, width, label=model,
                   color=colors[i % len(colors)], edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Held-out Module", fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f"Leave-One-Module-Out Transfer Performance ({metric})", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(modules)
        ax.legend(fontsize=11)
        plt.tight_layout()
        save_path = os.path.join(self.figure_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Transfer plot saved: {save_path}")

    def to_latex_table(
        self,
        df: pd.DataFrame,
        caption: str = "Model Comparison Results",
        label: str = "tab:results",
        save_name: str = "results_table.tex",
        bold_min_cols: List[str] = None,
        bold_max_cols: List[str] = None,
    ) -> str:
        """Generate LaTeX table from DataFrame with bolding of best values."""
        df_copy = df.copy()

        # Bold best values
        if bold_min_cols:
            for col in bold_min_cols:
                if col in df_copy.columns:
                    min_val = df_copy[col].min()
                    df_copy[col] = df_copy[col].apply(
                        lambda x: f"\\textbf{{{x:.4f}}}" if x == min_val else f"{x:.4f}"
                    )
        if bold_max_cols:
            for col in bold_max_cols:
                if col in df_copy.columns:
                    max_val = df_copy[col].max()
                    df_copy[col] = df_copy[col].apply(
                        lambda x: f"\\textbf{{{x:.4f}}}" if x == max_val else f"{x:.4f}"
                    )

        cols = " & ".join(df_copy.columns)
        col_fmt = "l" + "c" * (len(df_copy.columns) - 1)

        rows_str = ""
        for _, row in df_copy.iterrows():
            rows_str += " & ".join(str(v) for v in row.values) + " \\\\\n"

        latex = (
            f"\\begin{{table}}[htbp]\n"
            f"\\centering\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{col_fmt}}}\n"
            f"\\hline\n"
            f"{cols} \\\\\n"
            f"\\hline\n"
            f"{rows_str}"
            f"\\hline\n"
            f"\\end{{tabular}}\n"
            f"\\end{{table}}\n"
        )

        save_path = os.path.join(self.output_dir, save_name)
        with open(save_path, "w") as f:
            f.write(latex)
        logger.info(f"LaTeX table saved: {save_path}")
        return latex

    def generate_significance_report(
        self,
        sig_df: pd.DataFrame,
        save_name: str = "significance_report.csv",
    ):
        """Save significance test results."""
        path = os.path.join(self.output_dir, save_name)
        sig_df.to_csv(path, index=False)
        logger.info(f"Significance report saved: {path}")
        return path

    def plot_shap_modal_contribution(
        self,
        shap_tab: np.ndarray,
        feature_names: List[str],
        save_name: str = "shap_modal_contribution.png",
    ):
        """
        Compare SHAP contribution of behavioral features vs static features.
        """
        import matplotlib.pyplot as plt

        behavioral_kws = ["click", "week", "entropy", "growth", "ratio", "cv", "active"]
        behavioral_idx = [i for i, f in enumerate(feature_names)
                          if any(kw in f.lower() for kw in behavioral_kws)]
        static_idx = [i for i in range(len(feature_names)) if i not in behavioral_idx]

        sv = np.abs(np.array(shap_tab))
        behav_contrib = sv[:, behavioral_idx].sum(axis=1).mean() if behavioral_idx else 0
        static_contrib = sv[:, static_idx].sum(axis=1).mean() if static_idx else 0
        total = behav_contrib + static_contrib + 1e-10

        fig, ax = plt.subplots(figsize=(7, 5))
        labels = ["Behavioral\n(Sequential)", "Static\n(Demographic)"]
        values = [behav_contrib / total * 100, static_contrib / total * 100]
        colors = ["#4472C4", "#ED7D31"]
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black")

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")

        ax.set_ylabel("SHAP Contribution (%)", fontsize=13)
        ax.set_title("Modal Contribution Analysis (SHAP)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.2)
        plt.tight_layout()
        save_path = os.path.join(self.figure_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP modal contribution plot saved: {save_path}")
