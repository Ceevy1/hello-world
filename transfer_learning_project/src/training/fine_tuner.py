"""Fine-tuning strategies for transfer model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FineTuneConfig:
    strategy: str
    lr: float
    epochs: int
    unfreeze_interval: int = 5


def resolve_finetune_config(strategy: str, n_samples: int) -> FineTuneConfig:
    """Resolve strategy defaults for FT-1..FT-4."""
    if strategy == "FT-1":
        return FineTuneConfig(strategy=strategy, lr=1e-4, epochs=50)
    if strategy == "FT-2":
        return FineTuneConfig(strategy=strategy, lr=1e-3, epochs=30)
    if strategy == "FT-3":
        return FineTuneConfig(strategy=strategy, lr=3e-4, epochs=40, unfreeze_interval=5)
    if strategy == "FT-4":
        # Simplified proxy config for ProtoNet-style few-shot tuning
        epochs = 20 if n_samples < 50 else 30
        return FineTuneConfig(strategy=strategy, lr=5e-4, epochs=epochs)
    raise ValueError(f"Unknown finetune strategy: {strategy}")
