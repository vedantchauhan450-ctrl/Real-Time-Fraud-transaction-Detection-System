"""Anomaly detectors, scoring and evaluation helpers."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class PCADetector(BaseEstimator):
    """Reconstruction-error detector: fraud doesn't fit the 'normal' subspace."""
    def __init__(self, n_components=0.9):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.n_components, random_state=0).fit(X)
        return self

    def score_samples(self, X):  # higher = more normal (sklearn convention)
        rec = self.pca_.inverse_transform(self.pca_.transform(X))
        return -((np.asarray(X) - rec) ** 2).sum(axis=1)


def get_detectors(seed=42):
    return {
        "IsolationForest": IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1),
        "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1),
        "PCAReconstruction": PCADetector(0.9),
    }


def make_pipeline(detector):
    return Pipeline([("scale", RobustScaler()), ("det", detector)])


def anomaly_score(pipe, X):
    """Higher = more suspicious."""
    return -pipe.score_samples(X)


def threshold_from_normals(scores_normal, fpr=0.01):
    """Label-free threshold: flag at most `fpr` of legitimate traffic."""
    return float(np.quantile(scores_normal, 1 - fpr))


def precision_at_k(y, score, k):
    top = np.argsort(score)[::-1][:k]
    return float(np.asarray(y)[top].mean())


def evaluate(y, score, thr):
    y = np.asarray(y); pred = (score >= thr).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y, score),
        "PR_AUC": average_precision_score(y, score),
        "Prec@k": precision_at_k(y, score, int(y.sum())),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "Flagged%": pred.mean() * 100,
    }
