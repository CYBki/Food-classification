import sys
from pathlib import Path

# Ensure root path for module imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from helper_functions import accuracy_fn


def test_accuracy_fn_returns_100_when_preds_equal_labels():
    y_true = torch.tensor([1, 2, 3])
    y_pred = torch.tensor([1, 2, 3])
    acc = accuracy_fn(y_true, y_pred)
    assert acc == 100.0


def test_accuracy_fn_returns_0_when_preds_different_from_labels():
    y_true = torch.tensor([1, 2, 3])
    y_pred = torch.tensor([4, 5, 6])
    acc = accuracy_fn(y_true, y_pred)
    assert acc == 0.0
