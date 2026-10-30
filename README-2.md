# 🛡️ Real-Time Fraud Transaction Detection System

### Anomaly Detection on Credit Card Transactions with a Streaming Scoring Simulator

![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f89939?logo=scikit-learn&logoColor=white)
![Task](https://img.shields.io/badge/Task-Anomaly%20Detection-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

> An **unsupervised / semi-supervised** pipeline that learns what *normal* card behaviour looks like and flags transactions that deviate from it. A streaming simulator scores **one transaction at a time** and reports per-transaction latency and throughput, mimicking a real payment-authorisation flow.

---

## Table of Contents
1. [Overview](#overview)
2. [Why Anomaly Detection?](#why-anomaly-detection)
3. [Dataset](#dataset)
4. [System Architecture](#system-architecture)
5. [Project Structure](#project-structure)
6. [Methodology](#methodology)
7. [Evaluation](#evaluation)
8. [Quick Start](#quick-start)
9. [Real-Time Behaviour](#real-time-behaviour)
10. [Module Reference](#module-reference)
11. [Production Deployment Notes](#production-deployment-notes)
12. [Limitations](#limitations)
13. [Roadmap](#roadmap)
14. [FAQ](#faq)
15. [References](#references)

---

## Overview

Card fraud costs the industry billions every year, and every false alarm also costs money and customer trust. This project builds a complete detection pipeline: from raw transactions, through feature engineering and model training, to **real-time scoring** with risk bands and measurable latency.

Fraud is extremely rare (~0.17% of transactions) and its patterns change constantly, so labelled examples are scarce and go stale quickly. Instead of learning to *classify* fraud, we **model normality** and raise alerts on outliers.

**Key techniques demonstrated**

| Technique | Purpose |
|---|---|
| Train on legit-only data | Novelty detection: no fraud labels are needed to fit the model |
| Isolation Forest | Fast tree-based isolation of rare, easy-to-separate points |
| Local Outlier Factor (novelty mode) | Density-based detection of *local* anomalies |
| PCA reconstruction error | Fraud breaks the correlation structure that normal data follows |
| Robust scaling + `log(Amount)` | Resist heavy-tailed amounts and extreme outliers |
| Cyclic hour encoding (sin/cos) | Capture unusual time-of-day behaviour without a 23→0 discontinuity |
| Stratified split | Keep the rare fraud class represented in validation and test sets |
| Label-free threshold | Alert cut-off derived from a target false-alert rate (`--fpr`) |
| Validation-set grid search | Tune hyperparameters on PR-AUC, never on the test set |
| Streaming simulator | Per-transaction scoring with latency (mean / p95) and throughput |

---

## Why Anomaly Detection?

| Challenge | Why standard classifiers struggle | How this project handles it |
|---|---|---|
| **Extreme class imbalance** (~1 fraud in 600) | A model predicting "legit" always gets 99.8% accuracy | Metrics focus on PR-AUC and Precision@k; accuracy is not reported |
| **Label scarcity & delay** | Chargebacks arrive weeks later | Models fit on legit-only data; labels are used just for evaluation |
| **Concept drift** | Fraudsters change tactics constantly | "Normal" is a more stable target than "all possible fraud"; thresholds are recalibrated cheaply |
| **Novel attacks** | Supervised models only know past fraud | Outlier scores can flag never-seen patterns |
| **Latency budget** | Authorisation decisions take milliseconds | Lightweight models, one-row scoring, latency reporting |

---

## Dataset

**Credit Card Fraud Detection** (ULB / Worldline), available on Kaggle as `creditcard.csv`.

| Property | Value |
|---|---|
| Transactions | 284,807 over 2 days (European cardholders, September 2013) |
| Frauds | 492 (≈ 0.172%) |
| Features | `Time`, `V1`–`V28` (PCA-anonymised), `Amount` |
| Target | `Class` (1 = fraud, 0 = legit), used only for evaluation and threshold checks |

Place the file at `data/creditcard.csv`.

> `make_sample_data.py` creates a small **synthetic** file for smoke tests (checking that code runs). Never report metrics from synthetic data as real performance.

---

## System Architecture

```
                 ┌────────────────────────── OFFLINE (training) ──────────────────────────┐
 creditcard.csv ─► Stratified split ─► Legit-only subset ─► Feature pipeline ─► Detector fit │
                  (train/val/test)     (capped by           (log Amount,        (IForest /   │
                                        --max_train)         cyclic hour,        LOF / PCA)  │
                                                             robust scale)          │        │
                                       Validation scores ─► Threshold @ target FPR ◄┘        │
                 └──────────────────────────────────┬──────────────────────────────────────┘
                                                    │ saved: detector + threshold
                 ┌──────────────────────── ONLINE (scoring) ─────────────────────────────┐
 Transaction ───► Same feature pipeline ─► Anomaly score ─► score > threshold? ─► ALERT / OK
 (one at a time)                                              │                    + risk band
                                                              └─► latency & throughput log
                 └─────────────────────────────────────────────────────────────────────────┘
```

The **same preprocessing code** is used offline and online, which prevents training/serving skew.

---

## Project Structure

```
├── data/                 ← put creditcard.csv here
├── preprocessing.py      ← features, stratified split, legit-only subset
├── models.py             ← detectors, scoring, threshold, metrics
├── tuning.py             ← validation-set grid search (PR-AUC)
├── visualise.py          ← score distribution, PR / ROC curves
├── train.py              ← end-to-end CLI
├── stream.py             ← real-time simulation + latency report
├── predict.py            ← batch scoring with risk bands
├── make_sample_data.py   ← synthetic data for smoke tests only
├── outputs/              ← model_comparison.csv, predictions (auto-created)
├── figures/              ← generated plots (auto-created)
└── requirements.txt
```

---

## Methodology

### 1. Feature Engineering
| Feature | Treatment | Rationale |
|---|---|---|
| `Amount` | `log1p` transform, then robust scaling | Amounts are heavily right-skewed; log + median/IQR scaling limits the influence of extreme values |
| `Time` → hour of day | Cyclic encoding: `sin(2πh/24)`, `cos(2πh/24)` | 23:00 and 00:00 are close in behaviour, and a linear hour would put them far apart |
| `V1`–`V28` | Robust scaling | Already PCA components; scaling keeps distance-based models (LOF) fair |

### 2. Stratified Splitting & Legit-Only Training
Data is split with stratification so every split contains frauds. The detector is **fitted only on legitimate transactions** (capped at `--max_train` rows for speed). Frauds in validation and test sets are used purely for evaluation, mimicking real deployment where the model has never seen confirmed fraud.

### 3. Detectors

| Detector | Idea | Strengths | Weaknesses |
|---|---|---|---|
| **Isolation Forest** | Random trees isolate outliers in fewer splits, so a short average path = anomalous | Fast, scales well, little tuning | Can miss anomalies hidden inside dense regions |
| **Local Outlier Factor** (`novelty=True`) | Compares a point's local density with its neighbours' | Catches *local* anomalies that global methods miss | Slower at scoring, sensitive to `n_neighbors` |
| **PCA reconstruction error** | Project to top components and reconstruct; a large error means the point breaks the learned correlation structure | Very fast, interpretable (error per feature) | Assumes mostly linear structure |

All detectors expose a common interface so scores can be compared and thresholded identically (**higher score = more suspicious**).

### 4. Label-Free Threshold Selection
A fixed cut-off is chosen from the **distribution of legit-only validation scores**:

> `threshold = the (1 − fpr) quantile of legit scores`

With `--fpr 0.01`, roughly **1% of legitimate transactions** will be flagged. This ties the alert volume directly to analyst capacity and needs no fraud labels, which is important because labels arrive late in practice.

### 5. Hyperparameter Tuning
`--tune` runs a grid search (e.g. number of trees and contamination for Isolation Forest, `n_neighbors` for LOF, number of components for PCA), selecting the configuration with the best **validation PR-AUC**. The test set is never used for tuning.

### 6. Risk Bands (`predict.py`)
Batch scoring turns the continuous anomaly score into actionable bands relative to the threshold, so analysts can prioritise their queue instead of seeing a flat list of alerts.

---

## Evaluation

Accuracy is meaningless on data this imbalanced. We report:

| Metric | Why it matters |
|---|---|
| **PR-AUC** (primary) | Summarises precision/recall trade-off; honest under heavy imbalance |
| ROC-AUC | Threshold-free ranking quality (can look optimistic when negatives dominate) |
| **Precision@k** (k = number of frauds) | Of the k highest-scored transactions, how many are real fraud? Mirrors a fixed-size review queue |
| Precision / Recall / F1 @ threshold | Operational performance at the chosen `--fpr` |

### Results

> Run the pipeline on the real dataset and paste your numbers from `outputs/model_comparison.csv` below. Do **not** copy numbers from synthetic smoke-test data.

| Model | PR-AUC | ROC-AUC | Precision@k | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Isolation Forest | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| LOF (novelty) | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| PCA reconstruction | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

Generated figures (`figures/`): score distribution (legit vs fraud), Precision-Recall curve and ROC curve.

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. (Optional) smoke test without the Kaggle file
python make_sample_data.py

# 3. Train, tune and save the best detector + threshold
python train.py --data data/creditcard.csv --tune --fpr 0.01 --save_models

# 4. Real-time demo: score transactions one by one
python stream.py --input_csv data/creditcard.csv --n 2000

# 5. Batch scoring with risk bands
python predict.py --input_csv data/new_txns.csv
```

### CLI Options — `train.py`

| Flag | Default | Description |
|---|---|---|
| `--data` | `data/creditcard.csv` | Path to the dataset |
| `--fpr` | 0.01 | Target share of legit transactions that may be flagged |
| `--max_train` | 50000 | Cap on legit rows used for fitting (speed) |
| `--tune` | off | Grid search on validation PR-AUC |
| `--save_models` | off | Save best detector + threshold |

---

## Real-Time Behaviour

`stream.py` replays transactions **one at a time**, exactly as an authorisation system would see them:

1. Receive a single transaction.
2. Apply the saved feature pipeline.
3. Compute the anomaly score and compare with the saved threshold.
4. Emit an **ALERT** (or pass) and record how long the decision took.

At the end it prints the alert count, **mean and p95 latency per transaction**, and overall **throughput** (transactions/second). p95 matters more than the mean because slow tail requests are what breach real latency SLAs.

---

## Module Reference

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Load data, build features (log amount, cyclic hour, scaling), stratified split, legit-only subset |
| `models.py` | Detector wrappers (Isolation Forest, LOF, PCA), score function, threshold selection, metric computation |
| `tuning.py` | Validation-set grid search maximising PR-AUC |
| `visualise.py` | Score distribution, PR curve, ROC curve |
| `train.py` | End-to-end CLI: split → fit → tune → threshold → evaluate → save |
| `stream.py` | Per-transaction scoring simulator with latency and throughput report |
| `predict.py` | Batch scoring of new CSVs with risk bands |
| `make_sample_data.py` | Synthetic data generator (smoke tests only) |

---

## Production Deployment Notes

This repo is a simulator, not a production service. Recommended path to production:

- **REST service:** wrap the pipeline's `score_samples` in a FastAPI endpoint (`POST /score` → score, flag, band).
- **Streaming:** consume from a Kafka topic, score in micro-batches or per message, and publish alerts to a separate topic.
- **Monitoring:** track alert rate, score distribution drift and latency; a sudden shift in alert rate usually signals drift or a data-pipeline bug.
- **Recalibration:** recompute the threshold on fresh legit data on a schedule so the false-alert rate stays at the target.
- **Feedback loop:** feed analyst decisions and chargebacks back as labels, enabling a later supervised stage on top of the anomaly score.
- **Governance:** log every decision with model version and threshold for auditability.

---

## Limitations

- **Anonymised features** (`V1`–`V28`) limit interpretability; you can tell *that* a transaction is unusual but not *why* in business terms.
- **No per-card history:** the dataset lacks card/merchant IDs, so velocity features (transactions in the last hour, new-merchant flags, distance from last purchase) cannot be built.
- **Two-day window:** no weekly or seasonal patterns are visible.
- **Threshold drift:** behaviour changes over time, so thresholds must be re-calibrated regularly.
- **Unsupervised ceiling:** fully labelled supervised models typically score higher when plentiful, fresh labels exist; this project's strength is robustness when they don't.
- **Simulator latency** reflects one machine and Python overhead and is not a production benchmark.

---

## Roadmap

- [ ] Autoencoder detector (reconstruction error with a small neural network)
- [ ] Hybrid stage: supervised gradient boosting trained on top of anomaly scores
- [ ] Velocity / per-card behavioural features on datasets that include card IDs
- [ ] Drift detection (PSI / KS test) with automatic threshold recalibration
- [ ] FastAPI service + Docker image
- [ ] Kafka consumer demo with alert topic
- [ ] SHAP-style explanations for flagged transactions
- [ ] Cost-sensitive threshold using fraud loss vs review cost

---

## FAQ

**Why not just train a classifier like XGBoost?**
It is a strong option when you have plenty of recent labels. This project targets the realistic case where labels are rare and delayed, and it also catches *new* fraud types.

**What does `--fpr 0.01` actually mean?**
About 1% of normal transactions will trigger an alert. Lower it to reduce analyst workload at the cost of missing more fraud; raise it for the opposite.

**Why PR-AUC instead of ROC-AUC?**
With ~0.17% positives, ROC-AUC can look excellent even when most alerts are false. PR-AUC reflects the precision analysts actually experience.

**Can I use my own dataset?**
Yes. Provide a CSV with numeric features and an optional `Class` label, and adjust the feature lists in `preprocessing.py`.

---

## References

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM*.
- Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). LOF: Identifying Density-Based Local Outliers. *ACM SIGMOD*.
- Dal Pozzolo, A., et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE CIDM*.
- Saito, T., & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*, 10(3).
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

Built for the Anomaly Detection module · MIT License
