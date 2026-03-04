"""
Experiment Runner
Orchestrates all experiments:
  1. Standard model comparison (LSTM, XGB, CatBoost, Fusion)
  2. Early prediction (week4, week8, full)
  3. LOMO transfer learning
  4. MAML meta-learning
  5. SHAP interpretability
  6. Statistical significance tests
  7. Auto-generate figures and LaTeX tables
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

from config import (
    SEED, MODULES, LSTM_CONFIG, XGB_CONFIG, CATBOOST_CONFIG,
    FUSION_CONFIG, MAML_CONFIG, OUTPUT_DIR, MODEL_DIR, LOG_DIR, FIGURE_DIR
)
from data_preprocessing import (
    OULADDatasetBuilder, LeaveOneModuleOut, MetaTaskBuilder, split_dataset
)
from models import (
    LSTMTrainer, XGBoostModel, CatBoostModel,
    DynamicFusionTrainer, StackingFusion, MAMLTrainer
)
from evaluation import (
    compute_regression_metrics, compute_classification_metrics,
    SignificanceTester, SHAPAnalyzer, ResultsReporter
)


# ============================================================
# Helper: add file handler to logger
# ============================================================
def setup_file_logging(experiment_name: str):
    log_path = os.path.join(LOG_DIR, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
    logging.getLogger().addHandler(fh)
    return log_path


# ============================================================
# Experiment 1: Standard Model Comparison
# ============================================================
def run_standard_comparison(dataset: Dict, reporter: ResultsReporter) -> Dict:
    """
    Train LSTM, XGB, CatBoost, Stacking, and DynamicFusion.
    Evaluate on test set. Return predictions and metrics.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Standard Model Comparison")
    logger.info("=" * 60)

    train_ds, val_ds, test_ds = split_dataset(dataset, seed=SEED)

    X_seq_train  = train_ds["sequence"]
    X_tab_train  = train_ds["tabular"]
    y_train      = train_ds["y_reg"]

    X_seq_val    = val_ds["sequence"]
    X_tab_val    = val_ds["tabular"]
    y_val        = val_ds["y_reg"]

    X_seq_test   = test_ds["sequence"]
    X_tab_test   = test_ds["tabular"]
    y_test       = test_ds["y_reg"]

    results      = {}
    predictions  = {}

    # --- LSTM ---
    logger.info("Training LSTM...")
    lstm = LSTMTrainer(LSTM_CONFIG, input_dim=X_seq_train.shape[2], seed=SEED)
    lstm.fit(X_seq_train, y_train, X_seq_val, y_val)
    pred_lstm = lstm.predict(X_seq_test)
    results["LSTM"] = compute_regression_metrics(y_test, pred_lstm)
    predictions["LSTM"] = pred_lstm
    lstm.save(os.path.join(MODEL_DIR, "lstm.pt"))
    logger.info(f"LSTM: {results['LSTM']}")

    # --- XGBoost ---
    logger.info("Training XGBoost...")
    xgb_model = XGBoostModel(XGB_CONFIG)
    xgb_model.fit(X_tab_train, y_train, X_tab_val, y_val)
    pred_xgb = xgb_model.predict(X_tab_test)
    results["XGBoost"] = compute_regression_metrics(y_test, pred_xgb)
    predictions["XGBoost"] = pred_xgb
    xgb_model.save(os.path.join(MODEL_DIR, "xgboost.pkl"))
    logger.info(f"XGBoost: {results['XGBoost']}")

    # --- CatBoost ---
    logger.info("Training CatBoost...")
    cat_model = CatBoostModel(CATBOOST_CONFIG)
    cat_model.fit(X_tab_train, y_train, X_tab_val, y_val)
    pred_cat = cat_model.predict(X_tab_test)
    results["CatBoost"] = compute_regression_metrics(y_test, pred_cat)
    predictions["CatBoost"] = pred_cat
    cat_model.save(os.path.join(MODEL_DIR, "catboost.pkl"))
    logger.info(f"CatBoost: {results['CatBoost']}")

    # --- Stacking Fusion ---
    logger.info("Training Stacking Fusion...")
    train_base_preds = np.column_stack([
        lstm.predict(X_seq_train),
        xgb_model.predict(X_tab_train),
        cat_model.predict(X_tab_train),
    ])
    test_base_preds = np.column_stack([pred_lstm, pred_xgb, pred_cat])

    stacker = StackingFusion(meta_model_type=FUSION_CONFIG.get("meta_model", "linear"))
    stacker.fit(train_base_preds, y_train)
    pred_stack = stacker.predict(test_base_preds)
    results["Stacking"] = compute_regression_metrics(y_test, pred_stack)
    predictions["Stacking"] = pred_stack
    logger.info(f"Stacking: {results['Stacking']}")

    # --- Dynamic Weight Fusion ---
    logger.info("Training Dynamic Fusion...")
    dynamic_fuser = DynamicFusionTrainer(
        tabular_dim=X_tab_train.shape[1],
        config=FUSION_CONFIG,
        seed=SEED,
    )
    dynamic_fuser.fit(X_tab_train, train_base_preds, y_train)
    pred_dynamic, weights = dynamic_fuser.predict(X_tab_test, test_base_preds)
    results["DynamicFusion"] = compute_regression_metrics(y_test, pred_dynamic)
    predictions["DynamicFusion"] = pred_dynamic
    logger.info(f"DynamicFusion: {results['DynamicFusion']}")

    # --- Save and plot ---
    results_df = reporter.compile_metrics(results, "standard_comparison")
    reporter.plot_model_comparison(
        results_df, metric="RMSE",
        title="Model Comparison - RMSE",
        save_name="comparison_RMSE.png",
    )
    reporter.plot_model_comparison(
        results_df, metric="R2",
        title="Model Comparison - R²",
        save_name="comparison_R2.png",
    )
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
        "test_y": y_test,
        "models": {
            "lstm": lstm,
            "xgb": xgb_model,
            "cat": cat_model,
            "stacking": stacker,
            "dynamic": dynamic_fuser,
        },
        "test_datasets": {
            "X_seq": X_seq_test,
            "X_tab": X_tab_test,
        }
    }


# ============================================================
# Experiment 2: Early Prediction
# ============================================================
def run_early_prediction(reporter: ResultsReporter) -> Dict:
    """
    Train models on different prediction windows: week4, week8, full.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Early Prediction Experiment")
    logger.info("=" * 60)

    window_results = {}
    windows = {"week4": 4, "week8": 8, "full": None}

    for window_name, cutoff in windows.items():
        logger.info(f"--- Window: {window_name} ---")
        builder = OULADDatasetBuilder(cutoff_week=cutoff)
        dataset = builder.build()
        train_ds, val_ds, test_ds = split_dataset(dataset, seed=SEED)

        w_results = {}

        # XGBoost (fast baseline)
        xgb_m = XGBoostModel(XGB_CONFIG)
        xgb_m.fit(train_ds["tabular"], train_ds["y_reg"])
        pred = xgb_m.predict(test_ds["tabular"])
        w_results["XGBoost"] = compute_regression_metrics(test_ds["y_reg"], pred)

        # LSTM
        lstm_m = LSTMTrainer(LSTM_CONFIG, input_dim=dataset["sequence"].shape[2], seed=SEED)
        lstm_m.fit(train_ds["sequence"], train_ds["y_reg"], val_ds["sequence"], val_ds["y_reg"])
        pred_lstm = lstm_m.predict(test_ds["sequence"])
        w_results["LSTM"] = compute_regression_metrics(test_ds["y_reg"], pred_lstm)

        window_results[window_name] = w_results
        logger.info(f"Window {window_name}: {w_results}")

    reporter.plot_early_prediction_curve(
        window_results, metric="RMSE", save_name="early_prediction_RMSE.png"
    )
    reporter.plot_early_prediction_curve(
        window_results, metric="R2", save_name="early_prediction_R2.png"
    )

    # Flatten for latex
    rows = []
    for wname, wres in window_results.items():
        for model, metrics in wres.items():
            rows.append({"Window": wname, "Model": model, **metrics})
    early_df = pd.DataFrame(rows)
    early_df.to_csv(os.path.join(OUTPUT_DIR, "early_prediction_results.csv"), index=False)
    reporter.to_latex_table(
        early_df,
        caption="Early Prediction Performance Across Temporal Windows",
        label="tab:early_pred",
        save_name="early_prediction_table.tex",
        bold_min_cols=["RMSE", "MAE"],
        bold_max_cols=["R2"],
    )

    return window_results


# ============================================================
# Experiment 3: LOMO Transfer Learning
# ============================================================
def run_lomo_transfer(dataset: Dict, reporter: ResultsReporter) -> Dict:
    """
    Leave-One-Module-Out evaluation for transfer learning.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Leave-One-Module-Out Transfer Learning")
    logger.info("=" * 60)

    lomo = LeaveOneModuleOut()
    splits = lomo.split(dataset)
    lomo_results = {}

    for train_ds, test_ds, held_out_mod in splits:
        logger.info(f"Held-out module: {held_out_mod}")
        mod_results = {}

        # XGBoost
        xgb_m = XGBoostModel(XGB_CONFIG)
        xgb_m.fit(train_ds["tabular"], train_ds["y_reg"])
        pred = xgb_m.predict(test_ds["tabular"])
        mod_results["XGBoost"] = compute_regression_metrics(test_ds["y_reg"], pred)

        # CatBoost
        cat_m = CatBoostModel(CATBOOST_CONFIG)
        cat_m.fit(train_ds["tabular"], train_ds["y_reg"])
        pred_cat = cat_m.predict(test_ds["tabular"])
        mod_results["CatBoost"] = compute_regression_metrics(test_ds["y_reg"], pred_cat)

        # LSTM
        if len(train_ds["sequence"]) > 50:
            lstm_m = LSTMTrainer(LSTM_CONFIG, input_dim=dataset["sequence"].shape[2], seed=SEED)
            train_split = int(0.9 * len(train_ds["y_reg"]))
            lstm_m.fit(
                train_ds["sequence"][:train_split], train_ds["y_reg"][:train_split],
                train_ds["sequence"][train_split:], train_ds["y_reg"][train_split:],
            )
            pred_lstm = lstm_m.predict(test_ds["sequence"])
            mod_results["LSTM"] = compute_regression_metrics(test_ds["y_reg"], pred_lstm)

        lomo_results[held_out_mod] = mod_results
        logger.info(f"Module {held_out_mod}: {mod_results}")

    # Summary
    summary_rows = []
    for mod, res in lomo_results.items():
        for model, metrics in res.items():
            summary_rows.append({"Module": mod, "Model": model, **metrics})
    lomo_df = pd.DataFrame(summary_rows)
    lomo_df.to_csv(os.path.join(OUTPUT_DIR, "lomo_results.csv"), index=False)

    reporter.plot_transfer_comparison(lomo_results, metric="RMSE", save_name="lomo_transfer_RMSE.png")
    reporter.to_latex_table(
        lomo_df,
        caption="Leave-One-Module-Out Transfer Learning Results",
        label="tab:lomo",
        save_name="lomo_table.tex",
        bold_min_cols=["RMSE", "MAE"],
        bold_max_cols=["R2"],
    )

    # Summary statistics per model
    summary = lomo_df.groupby("Model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    logger.info(f"LOMO Summary:\n{summary}")
    summary.to_csv(os.path.join(OUTPUT_DIR, "lomo_summary.csv"))

    return lomo_results


# ============================================================
# Experiment 4: MAML Meta-Learning
# ============================================================
def run_maml(dataset: Dict, reporter: ResultsReporter) -> Dict:
    """
    MAML meta-learning across modules.
    Train on all modules, then fine-tune on each.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: MAML Meta-Learning")
    logger.info("=" * 60)

    task_builder = MetaTaskBuilder()
    tasks = task_builder.build_tasks(dataset)

    input_dim = dataset["tabular"].shape[1]
    maml_trainer = MAMLTrainer(input_dim, MAML_CONFIG, seed=SEED)

    # Meta-training
    logger.info("Meta-training MAML...")
    maml_trainer.meta_train(tasks)
    maml_trainer.save(os.path.join(MODEL_DIR, "maml_meta.pt"))

    # Fine-tuning and evaluation per task
    maml_results = {}
    for task in tasks:
        mod = task["module"]
        X = task["tabular"]
        y = task["y_reg"]
        n = len(y)

        if n < 20:
            logger.warning(f"Task {mod} too small ({n}), skipping.")
            continue

        # Support / query split (80/20 within task)
        split = int(n * 0.8)
        sup_x, qry_x = X[:split], X[split:]
        sup_y, qry_y = y[:split], y[split:]

        adapted = maml_trainer.fine_tune(sup_x, sup_y, n_steps=20)
        pred = maml_trainer.predict(adapted, qry_x)
        metrics = compute_regression_metrics(qry_y, pred)
        maml_results[mod] = metrics
        logger.info(f"MAML Module {mod}: {metrics}")

    # Save results
    maml_rows = [{"Module": mod, **metrics} for mod, metrics in maml_results.items()]
    maml_df = pd.DataFrame(maml_rows)
    maml_df.to_csv(os.path.join(OUTPUT_DIR, "maml_results.csv"), index=False)
    reporter.to_latex_table(
        maml_df,
        caption="MAML Meta-Learning Transfer Performance per Module",
        label="tab:maml",
        save_name="maml_table.tex",
        bold_min_cols=["RMSE", "MAE"],
        bold_max_cols=["R2"],
    )
    logger.info(f"MAML mean RMSE: {maml_df['RMSE'].mean():.4f} ± {maml_df['RMSE'].std():.4f}")
    return maml_results


# ============================================================
# Experiment 5: Statistical Significance Tests
# ============================================================
def run_significance_tests(
    predictions: Dict[str, np.ndarray],
    y_test: np.ndarray,
    reporter: ResultsReporter,
) -> pd.DataFrame:
    """
    Paired t-test and Wilcoxon signed-rank test.
    Compare DynamicFusion against all baselines.
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Statistical Significance Tests")
    logger.info("=" * 60)

    tester = SignificanceTester()
    sig_df = tester.compare_all(predictions, y_test, baseline="DynamicFusion")
    logger.info(f"Significance test results:\n{sig_df.to_string()}")
    reporter.generate_significance_report(sig_df, "significance_tests.csv")

    # Also generate LaTeX table
    sig_display = sig_df[["comparison", "t_pvalue", "t_label", "w_pvalue", "w_label"]].copy()
    sig_display.columns = ["Comparison", "t p-value", "t sig.", "Wilcoxon p-value", "W sig."]
    reporter.to_latex_table(
        sig_display,
        caption="Statistical Significance Tests (DynamicFusion vs Baselines)",
        label="tab:significance",
        save_name="significance_table.tex",
    )
    return sig_df


# ============================================================
# Experiment 6: SHAP Analysis
# ============================================================
def run_shap_analysis(
    xgb_model: XGBoostModel,
    X_tab: np.ndarray,
    feature_names,
    reporter: ResultsReporter,
):
    """Run SHAP analysis on XGBoost model."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 6: SHAP Interpretability Analysis")
    logger.info("=" * 60)

    if not isinstance(feature_names, list):
        feature_names = list(feature_names)

    analyzer = SHAPAnalyzer(feature_names)
    try:
        shap_vals = analyzer.explain_tree(xgb_model.model, X_tab)
    except Exception as e:
        logger.warning(f"Tree SHAP failed: {e}, using kernel SHAP.")
        shap_vals = analyzer.explain_kernel(xgb_model.predict, X_tab)

    if shap_vals is not None:
        # Global importance
        importance_df = analyzer.get_global_importance()
        importance_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)
        logger.info(f"Top 5 SHAP features:\n{importance_df.head()}")

        # Summary plot
        analyzer.plot_summary(save_path=os.path.join(FIGURE_DIR, "shap_summary.png"))

        # Modal contribution
        reporter.plot_shap_modal_contribution(
            np.array(shap_vals) if not isinstance(shap_vals, np.ndarray) else shap_vals,
            feature_names,
            save_name="shap_modal_contribution.png",
        )

        return importance_df
    return None


# ============================================================
# Main Runner
# ============================================================
def main():
    np.random.seed(SEED)

    log_path = setup_file_logging("full_experiment")
    logger.info(f"Experiment log: {log_path}")

    reporter = ResultsReporter()

    # ---- Load dataset (full period) ----
    logger.info("Building full dataset...")
    builder = OULADDatasetBuilder(cutoff_week=None)
    dataset = builder.build()

    N = len(dataset["y_reg"])
    T = dataset["sequence"].shape[1]
    D = dataset["sequence"].shape[2]
    F = dataset["tabular"].shape[1]
    logger.info(f"Dataset: N={N}, T={T}, D={D}, F={F}")

    # ---- Exp 1: Standard Comparison ----
    exp1 = run_standard_comparison(dataset, reporter)
    results    = exp1["results"]
    predictions = exp1["predictions"]
    y_test     = exp1["test_y"]
    models     = exp1["models"]
    test_data  = exp1["test_datasets"]

    # ---- Exp 2: Early Prediction ----
    window_results = run_early_prediction(reporter)

    # ---- Exp 3: LOMO Transfer ----
    lomo_results = run_lomo_transfer(dataset, reporter)

    # ---- Exp 4: MAML ----
    maml_results = run_maml(dataset, reporter)

    # ---- Exp 5: Significance Tests ----
    sig_df = run_significance_tests(predictions, y_test, reporter)

    # ---- Exp 6: SHAP ----
    feature_names = dataset.get("tab_feature_names", [f"feat_{i}" for i in range(F)])
    shap_result = run_shap_analysis(
        models["xgb"],
        test_data["X_tab"],
        feature_names,
        reporter,
    )

    # ---- Final Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE")
    logger.info("=" * 60)
    logger.info("\nMain Results:")
    for model, metrics in results.items():
        logger.info(f"  {model}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R2={metrics['R2']:.4f}")

    logger.info(f"\nOutputs saved to: {OUTPUT_DIR}")
    logger.info(f"Figures saved to: {FIGURE_DIR}")
    logger.info(f"Models saved to:  {MODEL_DIR}")
    return results


if __name__ == "__main__":
    main()
