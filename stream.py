"""Real-time simulation: score transactions ONE AT A TIME and raise alerts.
python stream.py --input_csv data/creditcard.csv --n 2000"""
import argparse, time
import joblib, numpy as np, pandas as pd
import preprocessing as pp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/best_model.joblib")
    ap.add_argument("--input_csv", default="data/creditcard.csv")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()

    b = joblib.load(a.model_path); pipe, thr = b["pipeline"], b["threshold"]
    df = pp.engineer_features(pp.load_data(a.input_csv)).sample(a.n, random_state=1)
    y = df.pop(pp.TARGET).values; X = df[b["features"]]

    lat, alerts, hits = [], 0, 0
    for i in range(len(X)):
        t0 = time.perf_counter()
        score = -pipe.score_samples(X.iloc[[i]])[0]          # one transaction
        lat.append((time.perf_counter() - t0) * 1000)
        if score >= thr:
            alerts += 1; hits += int(y[i])
            if alerts <= 5:
                print(f"ALERT txn#{i}  score={score:.2f}  actual={'FRAUD' if y[i] else 'legit'}")
    print(f"\nProcessed {len(X)} txns | alerts={alerts} | true frauds caught={hits}/{int(y.sum())}")
    print(f"Latency  mean={np.mean(lat):.2f} ms  p95={np.percentile(lat, 95):.2f} ms  "
          f"| throughput ~{1000 / np.mean(lat):.0f} txn/s")


if __name__ == "__main__":
    main()
