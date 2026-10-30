"""Grid search on the validation set (maximise PR-AUC; labels used only to tune)."""
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
import models as m

GRIDS = {
    "IsolationForest": {"det__n_estimators": [100, 300], "det__max_samples": [256, 1024, "auto"],
                        "det__max_features": [0.5, 1.0]},
    "LocalOutlierFactor": {"det__n_neighbors": [20, 35, 60]},
    "PCAReconstruction": {"det__n_components": [0.7, 0.8, 0.9, 0.95]},
}


def tune(name, pipe, X_fit, X_val, y_val):
    best, best_pipe = -1, None
    for params in ParameterGrid(GRIDS[name]):
        p = clone(pipe).set_params(**params).fit(X_fit)
        s = average_precision_score(y_val, m.anomaly_score(p, X_val))
        if s > best:
            best, best_pipe = s, p
    print(f"  tuned {name}: val PR-AUC={best:.4f}")
    return best_pipe
