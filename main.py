"""
Main Entry Point with Auto-Detected Backend
Selects PyTorch/XGBoost/CatBoost if available, else falls back to sklearn.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from config import (
    SEED, LSTM_CONFIG, XGB_CONFIG, CATBOOST_CONFIG,
    FUSION_CONFIG, MAML_CONFIG, OUTPUT_DIR, MODEL_DIR, LOG_DIR, FIGURE_DIR
)
from data_preprocessing import (
    OULADDatasetBuilder, LeaveOneModuleOut, MetaTaskBuilder, split_dataset
)
from evaluation import (
    compute_regression_metrics,
    SignificanceTester, SHAPAnalyzer, ResultsReporter
)

# --- Auto-detect available backends ---
TORCH_AVAILABLE = False
XGB_AVAILABLE = False
CAT_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    logger.info("PyTorch available.")
except ImportError:
    logger.info("PyTorch not available - using sklearn MLP as LSTM surrogate.")

try:
    import xgboost
    XGB_AVAILABLE = True
    logger.info("XGBoost available.")
except ImportError:
    logger.info("XGBoost not available - using GradientBoosting.")

try:
    import catboost
    CAT_AVAILABLE = True
    logger.info("CatBoost available.")
except ImportError:
    logger.info("CatBoost not available - using RandomForest.")

# Import appropriate model classes
if TORCH_AVAILABLE:
    from models import (
        LSTMTrainer, XGBoostModel, CatBoostModel,
        DynamicFusionTrainer, StackingFusion, MAMLTrainer
    )
    LSTMClass = LSTMTrainer
    XGBClass  = XGBoostModel
    CATClass  = CatBoostModel
    FusionClass = DynamicFusionTrainer
    StackClass = StackingFusion
    MAMLClass  = MAMLTrainer
else:
    from models_sklearn import (
        LSTMModelSklearn as LSTMClass,
        XGBoostModelSklearn as XGBClass,
        CatBoostModelSklearn as CATClass,
        DynamicFusionSklearn as FusionClass,
        StackingFusionSklearn as StackClass,
        MAMLSklearn as MAMLClass,
    )


# ============================================================
# Setup logging
# ============================================================
def setup_file_logging(name: str):
    path = os.path.join(LOG_DIR, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s"))
    logging.getLogger().addHandler(fh)
    return path


# ============================================================
# Experiment 1: Standard Model Comparison
# ============================================================
def run_standard_comparison(dataset: Dict, reporter: ResultsReporter) -> Dict:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Standard Model Comparison")
    logger.info("=" * 60)

    train_ds, val_ds, test_ds = split_dataset(dataset, seed=SEED)
    results, predictions = {}, {}

    # LSTM
    logger.info("Training LSTM / MLP-surrogate...")
    lstm = LSTMClass(LSTM_CONFIG, input_dim=dataset["sequence"].shape[2], seed=SEED)
    lstm.fit(train_ds["sequence"], train_ds["y_reg"], val_ds["sequence"], val_ds["y_reg"])
    pred_lstm = lstm.predict(test_ds["sequence"])
    results["LSTM"] = compute_regression_metrics(test_ds["y_reg"], pred_lstm)
    predictions["LSTM"] = pred_lstm
    lstm.save(os.path.join(MODEL_DIR, "lstm.pkl"))
    logger.info(f"LSTM: {results['LSTM']}")

    # XGBoost
    logger.info("Training XGBoost / GBM...")
    xgb = XGBClass(XGB_CONFIG)
    xgb.fit(train_ds["tabular"], train_ds["y_reg"])
    pred_xgb = xgb.predict(test_ds["tabular"])
    results["XGBoost"] = compute_regression_metrics(test_ds["y_reg"], pred_xgb)
    predictions["XGBoost"] = pred_xgb
    xgb.save(os.path.join(MODEL_DIR, "xgboost.pkl"))
    logger.info(f"XGBoost: {results['XGBoost']}")

    # CatBoost
    logger.info("Training CatBoost / RF...")
    cat = CATClass(CATBOOST_CONFIG)
    cat.fit(train_ds["tabular"], train_ds["y_reg"])
    pred_cat = cat.predict(test_ds["tabular"])
    results["CatBoost"] = compute_regression_metrics(test_ds["y_reg"], pred_cat)
    predictions["CatBoost"] = pred_cat
    cat.save(os.path.join(MODEL_DIR, "catboost.pkl"))
    logger.info(f"CatBoost: {results['CatBoost']}")

    # Stacking
    logger.info("Training Stacking Fusion...")
    train_base = np.column_stack([
        lstm.predict(train_ds["sequence"]),
        xgb.predict(train_ds["tabular"]),
        cat.predict(train_ds["tabular"]),
    ])
    test_base = np.column_stack([pred_lstm, pred_xgb, pred_cat])
    stacker = StackClass(meta_model_type="linear")
    stacker.fit(train_base, train_ds["y_reg"])
    pred_stack = stacker.predict(test_base)
    results["Stacking"] = compute_regression_metrics(test_ds["y_reg"], pred_stack)
    predictions["Stacking"] = pred_stack
    logger.info(f"Stacking: {results['Stacking']}")

    # Dynamic Fusion
    logger.info("Training Dynamic Fusion...")
    dynamic = FusionClass(tabular_dim=train_ds["tabular"].shape[1], config=FUSION_CONFIG, seed=SEED)
    dynamic.fit(train_ds["tabular"], train_base, train_ds["y_reg"])
    pred_dyn, weights = dynamic.predict(test_ds["tabular"], test_base)
    results["DynamicFusion"] = compute_regression_metrics(test_ds["y_reg"], pred_dyn)
    predictions["DynamicFusion"] = pred_dyn
    logger.info(f"DynamicFusion: {results['DynamicFusion']}")

    # Save
    results_df = reporter.compile_metrics(results, "standard_comparison")
    reporter.plot_model_comparison(results_df, "RMSE", "Model Comparison (RMSE)", "comparison_RMSE.png")
    reporter.plot_model_comparison(results_df, "MAE",  "Model Comparison (MAE)",  "comparison_MAE.png")
    reporter.plot_model_comparison(results_df, "R2",   "Model Comparison (R²)",   "comparison_R2.png")
    reporter.to_latex_table(
        results_df,
        caption="Performance Comparison of All Models on OULAD Dataset",
        label="tab:main_results",
        save_name="main_results.tex",
        bold_min_cols=["RMSE", "MAE"],
        bold_max_cols=["R2"],
    )

    return {
        "results": results,
        "predictions": predictions,
        "test_y": test_ds["y_reg"],
        "models": {"lstm": lstm, "xgb": xgb, "cat": cat},
        "test_data": {"X_seq": test_ds["sequence"], "X_tab": test_ds["tabular"]},
        "train_data": {"X_seq": train_ds["sequence"], "X_tab": train_ds["tabular"], "y": train_ds["y_reg"]},
    }


# ============================================================
# Experiment 2: Early Prediction
# ============================================================
def run_early_prediction(reporter: ResultsReporter) -> Dict:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Early Prediction")
    logger.info("=" * 60)

    windows = {"week4": 4, "week8": 8, "full": None}
    window_results = {}

    for wname, cutoff in windows.items():
        logger.info(f"--- Window: {wname} ---")
        ds = OULADDatasetBuilder(cutoff_week=cutoff).build()
        tr, va, te = split_dataset(ds, seed=SEED)

        wres = {}
        for ModelCls, mname, use_seq in [
            (LSTMClass, "LSTM",     True),
            (XGBClass,  "XGBoost",  False),
            (CATClass,  "CatBoost", False),
        ]:
            if use_seq:
                m = ModelCls(LSTM_CONFIG, input_dim=ds["sequence"].shape[2], seed=SEED)
                m.fit(tr["sequence"], tr["y_reg"], va["sequence"], va["y_reg"])
                pred = m.predict(te["sequence"])
            else:
                cfg = XGB_CONFIG if mname == "XGBoost" else CATBOOST_CONFIG
                m = ModelCls(cfg)
                m.fit(tr["tabular"], tr["y_reg"])
                pred = m.predict(te["tabular"])
            wres[mname] = compute_regression_metrics(te["y_reg"], pred)
        window_results[wname] = wres
        logger.info(f"  {wname}: {wres}")

    reporter.plot_early_prediction_curve(window_results, "RMSE", "early_prediction_RMSE.png")
    reporter.plot_early_prediction_curve(window_results, "R2",   "early_prediction_R2.png")

    rows = [{"Window": w, "Model": m, **met}
            for w, wr in window_results.items() for m, met in wr.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "early_prediction_results.csv"), index=False)
    reporter.to_latex_table(df, "Early Prediction Across Temporal Windows",
                            "tab:early", "early_prediction_table.tex",
                            bold_min_cols=["RMSE","MAE"], bold_max_cols=["R2"])
    return window_results


# ============================================================
# Experiment 3: LOMO Transfer
# ============================================================
def run_lomo_transfer(dataset: Dict, reporter: ResultsReporter) -> Dict:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: LOMO Transfer Learning")
    logger.info("=" * 60)

    lomo_results = {}
    for train_ds, test_ds, mod in LeaveOneModuleOut().split(dataset):
        mod_res = {}
        for ModelCls, mname, use_seq in [
            (XGBClass,  "XGBoost",  False),
            (CATClass,  "CatBoost", False),
            (LSTMClass, "LSTM",     True),
        ]:
            if use_seq:
                if len(train_ds["y_reg"]) < 50: continue
                m = ModelCls(LSTM_CONFIG, input_dim=dataset["sequence"].shape[2], seed=SEED)
                split = int(0.9 * len(train_ds["y_reg"]))
                m.fit(train_ds["sequence"][:split], train_ds["y_reg"][:split],
                      train_ds["sequence"][split:], train_ds["y_reg"][split:])
                pred = m.predict(test_ds["sequence"])
            else:
                cfg = XGB_CONFIG if mname == "XGBoost" else CATBOOST_CONFIG
                m = ModelCls(cfg)
                m.fit(train_ds["tabular"], train_ds["y_reg"])
                pred = m.predict(test_ds["tabular"])
            mod_res[mname] = compute_regression_metrics(test_ds["y_reg"], pred)
        lomo_results[mod] = mod_res
        logger.info(f"  Module {mod}: {mod_res}")

    rows = [{"Module": m, "Model": mn, **met}
            for m, mr in lomo_results.items() for mn, met in mr.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "lomo_results.csv"), index=False)
    reporter.plot_transfer_comparison(lomo_results, "RMSE", "lomo_transfer_RMSE.png")
    reporter.to_latex_table(df, "LOMO Transfer Learning Results", "tab:lomo",
                            "lomo_table.tex", bold_min_cols=["RMSE","MAE"], bold_max_cols=["R2"])

    summary = df.groupby("Model")[["RMSE","MAE","R2"]].agg(["mean","std"])
    logger.info(f"LOMO Summary:\n{summary}")
    summary.to_csv(os.path.join(OUTPUT_DIR, "lomo_summary.csv"))
    return lomo_results


# ============================================================
# Experiment 4: MAML
# ============================================================
def run_maml(dataset: Dict, reporter: ResultsReporter) -> Dict:
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: MAML Meta-Learning")
    logger.info("=" * 60)

    tasks = MetaTaskBuilder().build_tasks(dataset)
    maml = MAMLClass(dataset["tabular"].shape[1], MAML_CONFIG, seed=SEED)
    maml.meta_train(tasks)
    maml.save(os.path.join(MODEL_DIR, "maml.pkl"))

    maml_results = {}
    for task in tasks:
        mod = task["module"]
        n = len(task["y_reg"])
        if n < 20: continue
        split = int(n * 0.8)
        adapted = maml.fine_tune(task["tabular"][:split], task["y_reg"][:split], n_steps=20)
        pred = maml.predict(adapted, task["tabular"][split:])
        maml_results[mod] = compute_regression_metrics(task["y_reg"][split:], pred)
        logger.info(f"  MAML {mod}: {maml_results[mod]}")

    rows = [{"Module": m, **met} for m, met in maml_results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "maml_results.csv"), index=False)
    reporter.to_latex_table(df, "MAML Meta-Learning Performance", "tab:maml",
                            "maml_table.tex", bold_min_cols=["RMSE","MAE"], bold_max_cols=["R2"])
    logger.info(f"MAML mean RMSE: {df['RMSE'].mean():.4f} ± {df['RMSE'].std():.4f}")
    return maml_results


# ============================================================
# Experiment 5: Statistical Significance
# ============================================================
def run_significance_tests(predictions, y_test, reporter):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Statistical Significance Tests")
    logger.info("=" * 60)
    tester = SignificanceTester()
    sig_df = tester.compare_all(predictions, y_test, baseline="DynamicFusion")
    logger.info(f"\n{sig_df.to_string()}")
    reporter.generate_significance_report(sig_df, "significance_tests.csv")
    sig_display = sig_df[["comparison","t_pvalue","t_label","w_pvalue","w_label"]].copy()
    sig_display.columns = ["Comparison","t p-value","t sig.","Wilcoxon p-value","W sig."]
    reporter.to_latex_table(sig_display,
                            "Statistical Significance Tests (DynamicFusion vs Baselines)",
                            "tab:significance", "significance_table.tex")
    return sig_df


# ============================================================
# Experiment 6: SHAP
# ============================================================
def run_shap_analysis(xgb_model, X_tab, feature_names, reporter):
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: SHAP Interpretability")
    logger.info("=" * 60)
    if not isinstance(feature_names, list):
        feature_names = list(feature_names)

    analyzer = SHAPAnalyzer(feature_names)
    try:
        import shap
        shap_vals = analyzer.explain_tree(xgb_model.model, X_tab)
    except Exception:
        logger.warning("SHAP library unavailable. Using feature_importances_ as fallback.")
        fi = xgb_model.get_feature_importance()
        if fi is not None:
            df_fi = pd.DataFrame({"feature": feature_names[:len(fi)], "mean_abs_shap": fi})
            df_fi = df_fi.sort_values("mean_abs_shap", ascending=False)
            df_fi.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)
            logger.info(f"Top features (feature importance):\n{df_fi.head()}")
            # Plot feature importance
            _plot_feature_importance(df_fi, feature_names, reporter)
            return df_fi
        return None

    if shap_vals is not None:
        imp_df = analyzer.get_global_importance()
        imp_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)
        logger.info(f"Top SHAP features:\n{imp_df.head()}")
        analyzer.plot_summary(os.path.join(FIGURE_DIR, "shap_summary.png"))
        reporter.plot_shap_modal_contribution(
            np.array(shap_vals), feature_names, "shap_modal_contribution.png"
        )
        return imp_df
    return None


def _plot_feature_importance(df_fi: pd.DataFrame, feature_names, reporter):
    """Plot feature importance bar chart as SHAP fallback."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    top = df_fi.head(15)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#4472C4")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Feature Importance (Tree Model)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "feature_importance.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved: {path}")

    # Modal contribution fallback
    behavioral_kws = ["click", "week", "entropy", "growth", "ratio", "cv", "active"]
    all_feats = df_fi["feature"].tolist()
    b_sum = df_fi[df_fi["feature"].apply(
        lambda f: any(kw in f.lower() for kw in behavioral_kws))]["mean_abs_shap"].sum()
    s_sum = df_fi["mean_abs_shap"].sum() - b_sum
    total = b_sum + s_sum + 1e-10

    fig, ax = plt.subplots(figsize=(7, 5))
    vals = [b_sum / total * 100, s_sum / total * 100]
    bars = ax.bar(["Behavioral\n(Sequential)", "Static\n(Demographic)"],
                  vals, color=["#4472C4","#ED7D31"], width=0.5, edgecolor="black")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature Importance Contribution (%)", fontsize=13)
    ax.set_title("Modal Contribution Analysis", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(vals)*1.25)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "shap_modal_contribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(SEED)
    log_path = setup_file_logging("full_experiment")
    logger.info(f"Log: {log_path}")

    reporter = ResultsReporter()

    # Build dataset
    logger.info("Building dataset (synthetic if no data files found)...")
    dataset = OULADDatasetBuilder(cutoff_week=None).build()
    logger.info(f"Dataset: N={len(dataset['y_reg'])}, "
                f"T={dataset['sequence'].shape[1]}, "
                f"D={dataset['sequence'].shape[2]}, "
                f"F={dataset['tabular'].shape[1]}")

    # Run all experiments
    exp1 = run_standard_comparison(dataset, reporter)
    window_results = run_early_prediction(reporter)
    lomo_results   = run_lomo_transfer(dataset, reporter)
    maml_results   = run_maml(dataset, reporter)
    sig_df = run_significance_tests(exp1["predictions"], exp1["test_y"], reporter)
    feat_names = dataset.get("tab_feature_names", [f"feat_{i}" for i in range(dataset["tabular"].shape[1])])
    shap_result = run_shap_analysis(exp1["models"]["xgb"], exp1["test_data"]["X_tab"], feat_names, reporter)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("="*60)
    for model, m in exp1["results"].items():
        logger.info(f"  {model:15s}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")
    logger.info(f"\nOutputs: {OUTPUT_DIR}")
    logger.info(f"Figures: {FIGURE_DIR}")

    return exp1["results"]


if __name__ == "__main__":
    main()
