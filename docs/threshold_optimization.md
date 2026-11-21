# Threshold Optimization

## 1. Objective

Evaluate different cosine similarity thresholds to balance False Acceptance Rate (FAR) and False Rejection Rate (FRR).

## 2. Initial Custom Dataset Evaluation

### Methodology
The initial threshold scan uses `evaluation/results.csv`. Face detection and feature extraction are not repeated. Predictions are recalculated using different thresholds.

### Results

| Threshold | TP | FN | FP | TN | Accuracy | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 4 | 0 | 1 | 5 | 90% | 16.7% | 0% |
| 0.35 | 4 | 0 | 1 | 5 | 90% | 16.7% | 0% |
| 0.40 | 4 | 0 | 0 | 6 | 100% | 0% | 0% |
| 0.45 | 4 | 0 | 0 | 6 | 100% | 0% | 0% |
| 0.50 | 4 | 0 | 0 | 6 | 100% | 0% | 0% |
| 0.55 | 4 | 0 | 0 | 6 | 100% | 0% | 0% |
| 0.60 | 4 | 0 | 0 | 6 | 100% | 0% | 0% |
| 0.65 | 4 | 0 | 0 | 6 | 100% | 0% | 0% |
| 0.70 | 3 | 1 | 0 | 6 | 90% | 0% | 25% |

The highest unknown similarity was 0.378 and the lowest known similarity was 0.693.

## 3. Public Dataset Evaluation: LFW

### Dataset
Labeled Faces in the Wild (LFW).

- Total dataset identities: 5,749
- Total dataset images: 13,233
- Enrolled gallery identities: 100
- Known probe images evaluated: 226
- Unknown probe images evaluated: 100
- Skipped known probes: 2
- Skipped unknown probes: 0

### Experimental methodology

A custom open-set gallery/probe experiment was conducted using the YuNet face detector and SFace feature extractor.

One image per enrolled identity was used to construct the gallery. Other images belonging to enrolled identities were used as known probes. Images from identities excluded from the gallery were used as unknown probes.

Cosine similarity scores were compared against thresholds ranging from 0.30 to 0.70.

This is a custom LFW experiment and is NOT the official LFW 6,000-pair face verification benchmark.

### Threshold comparison

| Threshold | Accuracy | FAR | FRR | Exact ID Rate |
|---|---:|---:|---:|---:|
| 0.30 | 83.74% | 50.00% | 1.33% | 96.90% |
| 0.35 | 94.79% | 11.00% | 2.65% | 96.46% |
| 0.40 | 96.93% | 0.00% | 4.42% | 95.13% |
| 0.45 | 95.71% | 0.00% | 6.19% | 93.81% |
| 0.50 | 92.64% | 0.00% | 10.62% | 89.38% |
| 0.55 | 85.28% | 0.00% | 21.24% | 78.76% |
| 0.60 | 76.69% | 0.00% | 33.63% | 66.37% |
| 0.65 | 64.11% | 0.00% | 51.77% | 48.23% |
| 0.70 | 50.92% | 0.00% | 70.80% | 29.20% |

### Detailed metrics at threshold 0.45

| Metric | Result |
|---|---:|
| Binary acceptance accuracy | 95.71% |
| Precision | 100.00% |
| Recall | 93.81% |
| F1-score | 96.80% |
| FAR | 0.00% |
| FRR | 6.19% |
| Exact known identification rate | 93.81% |
| Known wrong identities | 0 |

### Threshold interpretation

At threshold 0.40:
- 216 known probes were accepted.
- 215 were assigned the correct identity.
- 1 was assigned an incorrect identity.
- 10 known probes were rejected.
- All 100 unknown probes were rejected.

At threshold 0.45:
- 212 known probes were accepted.
- All 212 were assigned the correct identity.
- 14 known probes were rejected.
- All 100 unknown probes were rejected.

Threshold 0.40 produced higher overall accuracy and F1-score, while threshold 0.45 produced no wrong identity assignments in this experiment.

## 4. Operating Threshold Decision

The project retains 0.45 as its current operating threshold.

This is a conservative preliminary decision that prioritizes avoiding incorrect identity assignments over accepting a few additional genuine probes.

Threshold 0.40 remains a promising alternative for further validation.

The threshold has not been independently validated on a separate held-out dataset. Therefore, neither value should be considered universally optimal.

## 5. Limitations

- The initial custom evaluation contains only 4 known and 6 unknown evaluated images.
- The LFW experiment uses a custom split, not the official 6,000-pair benchmark protocol.
- Only 100 enrolled and 100 unknown identities were selected for this experiment.
- Two known probe images were skipped.
- Results depend on the selected identities, images, gallery composition and threshold.
- These results do not establish real-world biometric accuracy, fairness or production security.

A separate held-out evaluation and broader testing are required before making stronger performance claims.

## 6. Visualization and Results

- Initial threshold visualization: `evaluation/threshold_analysis.png`
- LFW threshold sweep: `evaluation/lfw_results/lfw_threshold_sweep.csv`
- LFW detailed predictions: `evaluation/lfw_results/lfw_detailed_results.csv`
- LFW split manifest: `evaluation/lfw_results/lfw_split_manifest.csv`
- LFW summary: `evaluation/lfw_results/lfw_summary.json`
