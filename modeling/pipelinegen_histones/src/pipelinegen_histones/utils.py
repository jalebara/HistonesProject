import torch
import numpy as np

from sklearn.model_selection import GroupKFold


class RepeatedGroupKFold(GroupKFold):
    def __init__(self, n_splits=5, n_repeats=1):
        self.n_splits = n_splits
        self.n_repeats = n_repeats

    def split(self, X, y=None, groups=None):
        for _ in range(self.n_repeats):
            yield from super().split(X, y, groups)
