"""Batch scoring: python predict.py --input_csv data/new.csv"""
import argparse, os
import joblib, pandas as pd
import preprocessing as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/best_model.joblib")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", default="outputs/predictions.csv")
    a = ap.parse_args()

    b = joblib.load(a.model_path)
    X = pp.engineer_features(pp.load_data(a.input_csv)).drop(columns=[pp.TARGET], errors="ignore")[b["features"]]
    s = -b["pipeline"].score_samples(X)
    out = pd.DataFrame({"AnomalyScore": s.round(4), "Alert": (s >= b["threshold"]).astype(int)})
    out["Risk"] = pd.cut(s / b["threshold"], [-1e9, 1, 1.25, 1.6, 1e9], labels=["Normal", "Suspicious", "High", "Critical"])
    os.makedirs(os.path.dirname(a.output_csv) or ".", exist_ok=True); out.to_csv(a.output_csv, index=False)
    print(out.Risk.value_counts().to_string(), f"\nSaved -> {a.output_csv}")


if __name__ == "__main__":
    main()
