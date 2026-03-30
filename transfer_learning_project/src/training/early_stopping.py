"""Early stopping helper."""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        improved = score < self.best_score if self.mode == "min" else score > self.best_score
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
