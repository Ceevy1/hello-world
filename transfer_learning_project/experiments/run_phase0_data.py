"""Phase 0: data loading and summary."""

from __future__ import annotations

from pathlib import Path

from src.data.data_loader import DataLoader, save_data_report


def run_phase0(config: dict, logger) -> None:
    loader = DataLoader(config.get("custom_dataset", {}).get("columns", {}))
    target_path = config["data"]["target"]["path"]
    if not Path(target_path).exists():
        logger.warning("Target dataset not found, skip phase0 report generation")
        return
    df = loader.load_custom_dataset(target_path, anonymize=True)
    report = loader.generate_data_report(df)
    save_data_report(report, "data/processed/data_summary.json")
    logger.info("Phase0 complete: data_summary.json generated")
