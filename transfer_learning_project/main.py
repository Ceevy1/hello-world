#!/usr/bin/env python3
"""多源特征动态策略学业成绩预测框架 - 迁移学习实验主入口。"""

from __future__ import annotations

import argparse

import yaml

from src.utils.logger import setup_logger
from src.utils.seed_manager import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer Learning Experiment")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--phase", type=str, default="all", help="运行阶段：all / 1 / 2,3 / 0-5")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复实验")
    return parser.parse_args()


def parse_phase_arg(phase_arg: str) -> list[int]:
    if phase_arg == "all":
        return [0, 1, 2, 3, 4, 5, 6]
    if "-" in phase_arg:
        start, end = map(int, phase_arg.split("-"))
        return list(range(start, end + 1))
    return [int(x.strip()) for x in phase_arg.split(",") if x.strip()]


def run_experiment(config: dict, phases: list[int], debug: bool = False) -> None:
    logger = setup_logger(config, debug)
    set_global_seed(config["project"]["seed"])

    if 0 in phases:
        from experiments.run_phase0_data import run_phase0

        run_phase0(config, logger)
    if 1 in phases:
        from experiments.run_phase1_feature import run_phase1

        run_phase1(config, logger)
    if 2 in phases:
        from experiments.run_phase2_align import run_phase2

        run_phase2(config, logger)
    if 3 in phases:
        from experiments.run_phase3_finetune import run_phase3

        run_phase3(config, logger)
    if 4 in phases:
        from experiments.run_phase4_baseline import run_phase4

        run_phase4(config, logger)
    if 5 in phases:
        from experiments.run_phase5_ablation import run_phase5

        run_phase5(config, logger)
    if 6 in phases:
        from experiments.run_phase6_generalization import run_phase6

        run_phase6(config, logger)


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    phases = parse_phase_arg(args.phase)
    run_experiment(cfg, phases, args.debug)
