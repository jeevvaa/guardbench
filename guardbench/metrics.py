from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, y_true: int, y_pred: int) -> None:
        if y_true == 1 and y_pred == 1:
            self.tp += 1
        elif y_true == 0 and y_pred == 1:
            self.fp += 1
        elif y_true == 0 and y_pred == 0:
            self.tn += 1
        else:
            self.fn += 1

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0

    def fnr(self) -> float:
        d = self.fn + self.tp
        return self.fn / d if d else 0.0

    def accuracy(self) -> float:
        d = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / d if d else 0.0
