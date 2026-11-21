# Threshold Optimization

## Objective
Evaluate different cosine similarity thresholds to balance False Acceptance Rate (FAR) and False Rejection Rate (FRR).

## Methodology
The threshold scan uses the existing evaluation/results.csv file. Face detection and feature extraction are not repeated. Predictions are recalculated using different thresholds.

## Results
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

## Decision
The project retains a cosine similarity threshold of 0.45. In this small test dataset, the highest unknown similarity was 0.378 and the lowest known similarity was 0.693.

## Visualization
See `threshold_analysis.png` for the FAR vs FRR threshold plot.

## Limitation
These results are based on only 4 known and 6 unknown evaluated images. The 100% accuracy observed here does not establish real-world performance or generalization. A larger and more diverse dataset is required.
