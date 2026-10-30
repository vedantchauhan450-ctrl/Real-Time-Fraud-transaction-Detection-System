"""python train.py --data data/creditcard.csv --tune --save_models"""
import argparse, os
import joblib, pandas as pd
import preprocessing as pp, models as m, tuning, visualise as viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--fpr", type=float, default=0.01, help="target false-alert rate on legit traffic")
    ap.add_argument("--max_train", type=int, default=50000, help="cap on normal rows used to fit")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--no_plots", action="store_true")
    a = ap.parse_args()

    df = pp.engineer_features(pp.load_data(a.data))
    X_tr, X_va, X_te, y_tr, y_va, y_te = pp.split_data(df)
    X_fit = pp.normal_only(X_tr, y_tr, a.max_train)
    print(f"Fit on {len(X_fit)} legit txns | test {len(X_te)} ({y_te.sum()} fraud)")

    fitted, rows, test_scores = {}, [], {}
    for name, det in m.get_detectors().items():
        pipe = m.make_pipeline(det)
        pipe = tuning.tune(name, pipe, X_fit, X_va, y_va) if a.tune else pipe.fit(X_fit)
        thr = m.threshold_from_normals(m.anomaly_score(pipe, X_va[y_va == 0]), a.fpr)
        s = m.anomaly_score(pipe, X_te)
        fitted[name], test_scores[name] = (pipe, thr), s
        rows.append({"Model": name, **m.evaluate(y_te, s, thr)})

    res = pd.DataFrame(rows).set_index("Model").round(4).sort_values("PR_AUC", ascending=False)
    print("\n", res.to_string())
    os.makedirs("outputs", exist_ok=True); res.to_csv("outputs/model_comparison.csv")

    best = res.index[0]; pipe, thr = fitted[best]; print(f"\nBest detector: {best}")
    if a.save_models:
        os.makedirs(a.models_dir, exist_ok=True)
        joblib.dump({"pipeline": pipe, "threshold": thr, "name": best,
                     "features": list(X_te.columns)}, f"{a.models_dir}/best_model.joblib")
        print(f"Saved -> {a.models_dir}/best_model.joblib")
    if not a.no_plots:
        viz.plot_curves(test_scores, y_te)
        viz.plot_score_dist(test_scores[best], y_te, thr, best)
        print("Figures saved in figures/")


if __name__ == "__main__":
    main()
