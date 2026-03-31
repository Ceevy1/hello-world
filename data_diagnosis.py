#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from paper_generalization.common import TARGET_COL, infer_features, load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    df = load_dataset(args.input)
    feature_cols = infer_features(df)

    report = []
    report.append(f"样本量: {len(df)}")
    report.append(f"列数: {df.shape[1]}")
    report.append(f"目标列: {TARGET_COL}")
    report.append("\n缺失率:")
    report.append(df.isna().mean().to_string())
    report.append("\n重复列:")
    report.append(str(df.columns[df.columns.duplicated()].tolist()))
    report.append("\n数值特征方差:")
    report.append(df[feature_cols].var().sort_values(ascending=False).to_string())
    report.append("\n目标统计:")
    report.append(df[TARGET_COL].describe().to_string())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"saved diagnosis to {out}")


if __name__ == "__main__":
    main()
