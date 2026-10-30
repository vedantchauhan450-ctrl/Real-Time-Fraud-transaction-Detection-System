# 🛡️ Real-Time Fraud Detection System

### Anomaly Detection on Credit Card Transactions

> An unsupervised / semi-supervised pipeline that learns what *normal* card behaviour looks like and flags transactions that deviate from it, with a streaming simulator that scores one transaction at a time and reports latency.

## Overview
Fraud is extremely rare (~0.17% of transactions) and constantly changing, so labelled examples are scarce and quickly outdated. Instead of classifying, we **model normality** and raise alerts on outliers.

| Technique | Purpose |
|---|---|
| Train on legit-only data | Novelty detection; no fraud labels needed to fit |
| Isolation Forest | Fast tree-based isolation of rare points |
| Local Outlier Factor (novelty) | Density-based local anomalies |
| PCA reconstruction error | Fraud breaks the correlation structure of normal data |
| Robust scaling + log(Amount) | Resist heavy-tailed amounts and outliers |
| Cyclic hour encoding | Capture unusual-time-of-day behaviour |
| Label-free threshold | Alert cut-off set from a target false-alert rate (`--fpr`) |
| Streaming simulator | Per-transaction scoring, latency & throughput |

## Dataset
**Credit Card Fraud Detection** (ULB / Worldline, Kaggle `creditcard.csv`) - 284,807 transactions over 2 days, 492 frauds. Features: `Time`, `V1-V28` (PCA-anonymised), `Amount`, target `Class`. Place at `data/creditcard.csv`.

## Project Structure
```
├── data/                 ← put creditcard.csv here
├── preprocessing.py      ← features, stratified split, legit-only subset
├── models.py             ← detectors, scoring, threshold, metrics
├── tuning.py             ← validation-set grid search (PR-AUC)
├── visualise.py          ← score distribution, PR/ROC curves
├── train.py              ← end-to-end CLI
├── stream.py             ← real-time simulation + latency report
├── predict.py            ← batch scoring with risk bands
├── make_sample_data.py   ← synthetic data for smoke tests only
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt
python train.py --data data/creditcard.csv --tune --fpr 0.01 --save_models
python stream.py --input_csv data/creditcard.csv --n 2000     # real-time demo
python predict.py --input_csv data/new_txns.csv               # batch scoring
```

| Flag | Default | Description |
|---|---|---|
| `--fpr` | 0.01 | Target share of legit transactions that may be flagged |
| `--max_train` | 50000 | Cap on legit rows used for fitting (speed) |
| `--tune` | off | Grid search on validation PR-AUC |
| `--save_models` | off | Save best detector + threshold |

## Evaluation
Accuracy is meaningless here. We report **PR-AUC** (primary), ROC-AUC, Precision@k (k = number of frauds), plus Precision/Recall/F1 at the chosen threshold. Paste your results from `outputs/model_comparison.csv` here after running on the real dataset.

## Real-Time Behaviour
`stream.py` scores each transaction individually and prints alerts plus mean/p95 latency and throughput. For production, wrap `pipeline.score_samples` in a REST service (FastAPI) or a Kafka consumer.

## Limitations
Anonymised features limit interpretability; no per-card history (velocity features) is available in this dataset; thresholds should be re-calibrated as behaviour drifts.

## References
- Liu, Ting & Zhou (2008). Isolation Forest. *IEEE ICDM*.
- Breunig et al. (2000). LOF: Identifying Density-Based Local Outliers. *ACM SIGMOD*.
- Dal Pozzolo et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE CIDM*.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12.

Built for the Anomaly Detection module · MIT License
