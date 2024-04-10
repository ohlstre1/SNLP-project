# Author: Linus Lind
# Date: 21 March 2024 
# LICENSE: GNU GPLv3
###############################################################################
# Metrics for calculating precision, recall, f1-score, f beta score and accuracy,
# uses vectorized numpy operations 
import numpy as np

def precision(pred: np.ndarray, 
              real: np.ndarray) -> float:
    ones_pred = np.equal(pred, 1)
    ones_real = np.equal(real, 1)
    zeros_real = np.equal(real, 0)
    TP = np.sum(np.where(np.logical_and(ones_pred, ones_real), 1, 0))
    FP = np.sum(np.where(np.logical_and(ones_pred, zeros_real), 1, 0))
    return TP / max((TP + FP), 1)

def recall(pred: np.ndarray, 
           real: np.ndarray) -> float:
    ones_pred = np.equal(pred, 1)
    ones_real = np.equal(real, 1)
    zeros_pred = np.equal(pred, 0)
    TP = np.sum(np.where(np.logical_and(ones_pred, ones_real), 1, 0))
    FN = np.sum(np.where(np.logical_and(zeros_pred, ones_real), 1, 0))
    return TP / max((TP + FN), 1)

def f1_score(pred: np.ndarray, 
             real: np.ndarray) -> float:
    prec = precision(pred, real)
    rec = recall(pred, real)
    return (2 * prec * rec) / max((prec + rec), 1)

def fx_score(pred: np.ndarray, 
             real: np.ndarray, 
             beta: float) -> float:
    prec = precision(pred, real)
    rec = recall(pred, real)
    return (1+beta**2) * (prec * rec / ((beta**2 *prec) + rec))

def accuracy(pred: np.ndarray, 
             real: np.ndarray) -> float:
    ones_pred = np.equal(pred, 1)
    ones_real = np.equal(real, 1)
    zeros_pred = np.equal(pred, 0)
    zeros_real = np.equal(real, 0)
    TP = np.sum(np.where(np.logical_and(ones_pred, ones_real), 1, 0))
    TN = np.sum(np.where(np.logical_and(zeros_pred, zeros_real), 1, 0))
    N = len(pred)
    return (TP + TN) / max(N, 1)
