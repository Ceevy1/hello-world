from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import DataLoader

from data_loader import OULADDataset, construct_knowledge_graph, extract_time_series, load_junyi_data, load_oulad_data
from experiment.evaluation.metrics import evaluate
from experiment.evaluation.significance_test import paired_t_test
from experiment.results.latex_generator import to_latex
from experiment.results.save_results import save_results
from experiment.splits.lomo_split import lomo_split
from feature_engineering import extract_features
from model import DynamicFusionEnhanced
from trainer import TrainConfig, fit


def _window_seq(x_seq: np.ndarray, week: int, max_weeks: int = 16) -> tuple[np.ndarray, np.ndarray]:
    out = np.copy(x_seq)
    out[:, week:, :] = 0.0
    return out, np.full(len(x_seq), min(week - 1, max_weeks - 1), dtype=np.int64)


def _predict_dynamic(model: DynamicFusionEnhanced, loader: DataLoader, graph, device="cpu") -> np.ndarray:
    import torch

    model.eval()
    node_features, edge_index = graph
    node_features = node_features.to(device)
    edge_index = edge_index.to(device)
    pred_all = []
    with torch.no_grad():
        for batch in loader:
            x_seq = np.stack([b.x_seq.numpy() for b in batch])
            x_stat = np.stack([b.x_stat.numpy() for b in batch])
            node_idx = np.stack([b.node_index.numpy() for b in batch])
            week = np.stack([b.week_index.numpy() for b in batch])
            x_seq_t = torch.tensor(x_seq, dtype=torch.float32, device=device)
            x_stat_t = torch.tensor(x_stat, dtype=torch.float32, device=device)
            node_idx_t = torch.tensor(node_idx, dtype=torch.long, device=device)
            week_t = torch.tensor(week, dtype=torch.long, device=device)
            p, _, _ = model(x_seq_t, x_stat_t, node_idx_t, week_t, node_features, edge_index)
            pred_all.append(p.cpu().numpy())
    return np.concatenate(pred_all)


def _train_dynamic_split(x_seq_train, x_stat_train, y_train, week_train, x_seq_test, x_stat_test, y_test, week_test, seed: int, epochs: int):
    import torch

    x_tr, x_va, s_tr, s_va, y_tr, y_va, w_tr, w_va = train_test_split(
        x_seq_train, x_stat_train, y_train, week_train, test_size=0.15, random_state=seed, stratify=y_train
    )

    n_tr, n_va, n_te = len(y_tr), len(y_va), len(y_test)
    node_all = np.arange(n_tr + n_va + n_te, dtype=np.int64)

    tr_ds = OULADDataset(x_tr, s_tr, node_all[:n_tr], w_tr, y_tr)
    va_ds = OULADDataset(x_va, s_va, node_all[n_tr:n_tr + n_va], w_va, y_va)
    te_ds = OULADDataset(x_seq_test, x_stat_test, node_all[n_tr + n_va:], week_test, y_test)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, collate_fn=list)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, collate_fn=list)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False, collate_fn=list)

    graph = construct_knowledge_graph(num_students=n_tr + n_va + n_te)
    model = DynamicFusionEnhanced(seq_input_dim=x_seq_train.shape[-1], stat_input_dim=x_stat_train.shape[-1], graph_input_dim=16)

    tmp_dir = Path("outputs/.tmp_generalization")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fit(model, tr_loader, va_loader, graph, TrainConfig(epochs=epochs), output_dir=str(tmp_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    prob = _predict_dynamic(model, te_loader, graph, device=device)
    return evaluate(y_test, prob), prob


def run_lomo_experiment(x_seq, x_stat, y, modules, week_idx, seeds: list[int], week_name: str, week_len: int, epochs: int):
    results = []
    splits = lomo_split(modules)
    x_seq_w, week_w = _window_seq(x_seq, week_len)

    for split in splits:
        tr, te = split["train_mask"], split["test_mask"]
        if tr.sum() < 50 or te.sum() < 20 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue

        acc_runs, auc_runs = [], []
        for sd in seeds:
            m, _ = _train_dynamic_split(x_seq_w[tr], x_stat[tr], y[tr], week_w[tr], x_seq_w[te], x_stat[te], y[te], week_w[te], sd, epochs)
            acc_runs.append(m["Accuracy"])
            auc_runs.append(m["AUC"])

        results.append({
            "模块ID": split["test_module"],
            "样本量": int(te.sum()),
            f"{week_name}_Acc": float(np.mean(acc_runs)),
            f"{week_name}_AUC": float(np.mean(auc_runs)),
            f"{week_name}_Acc_std": float(np.std(acc_runs)),
            f"{week_name}_AUC_std": float(np.std(auc_runs)),
        })
        print(f"[LOMO][{week_name}] module={split['test_module']} Acc={np.mean(acc_runs):.4f}±{np.std(acc_runs):.4f} AUC={np.mean(auc_runs):.4f}±{np.std(auc_runs):.4f}")

    return results


def run_cross_dataset(x_seq_o, x_stat_o, y_o, x_seq_j, x_stat_j, y_j, epochs: int, seed: int):
    # simplified aligned tabular baseline + dynamic as reference split
    x_o = extract_features(x_seq_o, x_stat_o)
    x_j = extract_features(x_seq_j, x_stat_j)

    base = LogisticRegression(max_iter=500)
    base.fit(x_o, y_o)
    pred_oj = base.predict_proba(x_j)[:, 1]
    met_oj = evaluate(y_j, pred_oj)

    base2 = LogisticRegression(max_iter=500)
    base2.fit(x_j, y_j)
    pred_jo = base2.predict_proba(x_o)[:, 1]
    met_jo = evaluate(y_o, pred_jo)

    print(f"[CrossDataset] OULAD->Junyi Acc={met_oj['Accuracy']:.4f} AUC={met_oj['AUC']:.4f}")
    print(f"[CrossDataset] Junyi->OULAD Acc={met_jo['Accuracy']:.4f} AUC={met_jo['AUC']:.4f}")
    return [
        {"训练数据": "OULAD", "测试数据": "Junyi", **met_oj},
        {"训练数据": "Junyi", "测试数据": "OULAD", **met_jo},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/experiment")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", default="42,52,62,72,82")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    x_seq, x_stat, _, week_idx, y, modules, _ = extract_time_series(load_oulad_data(args.data_dir), max_weeks=16)
    j_seq, j_stat, _, _, j_y, _, _ = load_junyi_data(None, max_weeks=16)

    # LOMO results per early window
    rows = []
    for name, week in [("4周", 4), ("8周", 8), ("全程", 16)]:
        rows.extend(run_lomo_experiment(x_seq, x_stat, y, modules, week_idx, seeds, name, week, args.epochs))
    lomo_df = pd.DataFrame(rows)
    save_results(lomo_df, str(out / "oulad_LOMO_results.csv"))

    # performance drop (in-domain as module mean at 全程, lomo as mean full)
    if not lomo_df.empty and "全程_Acc" in lomo_df.columns:
        in_domain = float(lomo_df["全程_Acc"].max())
        lomo_mean = float(lomo_df["全程_Acc"].mean())
        drop = (in_domain - lomo_mean) / max(in_domain, 1e-8)
        pd.DataFrame([{"in_domain_acc": in_domain, "lomo_acc": lomo_mean, "generalization_drop": drop}]).to_csv(out / "generalization_drop.csv", index=False)
        print(f"[GeneralizationDrop] in-domain={in_domain:.4f}, lomo={lomo_mean:.4f}, drop={drop:.4f}")

    cross_rows = run_cross_dataset(x_seq, x_stat, y, j_seq, j_stat, j_y, args.epochs, seeds[0])
    cross_df = pd.DataFrame(cross_rows)
    save_results(cross_df, str(out / "cross_dataset_results.csv"))

    # t-test: OULAD->Junyi vs Junyi->OULAD over metrics vector
    t = paired_t_test(
        np.array([cross_df.iloc[0]["Accuracy"], cross_df.iloc[0]["AUC"], cross_df.iloc[0]["F1"]]),
        np.array([cross_df.iloc[1]["Accuracy"], cross_df.iloc[1]["AUC"], cross_df.iloc[1]["F1"]]),
    )
    tt_df = pd.DataFrame([{"Model": "OULAD->Junyi", "Baseline": "Junyi->OULAD", **t}])
    tt_df.to_csv(out / "significance_ttest.csv", index=False)
    print(f"[TTest] p-value={t['p_value']:.6f}, significant={t['significant']}")

    # quick plot
    if not lomo_df.empty and "全程_Acc" in lomo_df.columns:
        plot_df = lomo_df[["模块ID", "全程_Acc"]].dropna()
        if not plot_df.empty:
            plt.figure(figsize=(8, 4))
            plt.bar(plot_df["模块ID"].astype(str), plot_df["全程_Acc"])
            plt.xticks(rotation=45, ha="right")
            plt.title("LOMO Full Accuracy by Module")
            plt.tight_layout()
            plt.savefig(out / "generalization_comparison.png", dpi=220)
            plt.close()

    (out / "generalization_table.tex").write_text(to_latex(lomo_df.fillna("") if not lomo_df.empty else pd.DataFrame()), encoding="utf-8")


if __name__ == "__main__":
    main()
