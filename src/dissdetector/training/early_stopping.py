import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported EarlyStopping mode: {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = None
        self.counter = 0
        self.should_stop = False

    def step(self, current_value: float) -> None:
        if np.isnan(current_value):
            return

        if self.best_value is None:
            self.best_value = current_value
            self.counter = 0
            return

        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.should_stop = True
