"""Confusion matrices and mIoU, computed as in HyperFLAW's metrics/stream_metrics.py.

Labels outside [0, 19) (255 = ignored) are skipped. IoU = TP / (TP + FP + FN + 1e-6); mIoU is the mean over the classes
present in the labels of the matrix (so 19 classes for the street images, 5 for the drone images).
"""
import numpy as np

N_CLASSES = 19


def confusion(label, pred, n=N_CLASSES):
    """(n, n) int64 matrix, rows = label, columns = prediction."""
    label, pred = np.asarray(label).ravel(), np.asarray(pred).ravel()
    m = (label >= 0) & (label < n)
    return np.bincount(n * label[m].astype(np.int64) + pred[m], minlength=n * n).reshape(n, n)


def iou(cm):
    cm = np.asarray(cm, dtype=np.float64)
    diag = np.diag(cm)
    return diag / (cm.sum(1) + cm.sum(0) - diag + 1e-6), cm.sum(1) != 0


def miou(cm):
    """mIoU (%) over the classes present in the labels."""
    v, present = iou(cm)
    return float(100 * v[present].mean())


def class_iou(cm, names):
    """{name: IoU (%)} for the classes in `names` ({class id: name}) present in the labels."""
    v, present = iou(cm)
    return {name: float(100 * v[c]) for c, name in names.items() if present[c]}
