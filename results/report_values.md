# Results for report

All scores use the 0–1 scale. Main and error-type tables show Macro-F1.

## Main results

| Model | 0% | 5% | 10% | 20% | 30% |
|---|---:|---:|---:|---:|---:|
| Word TF-IDF + LR | 0.9208 | 0.9189 | 0.9151 | 0.9113 | 0.9022 |
| Char TF-IDF + LR | 0.9204 | 0.9193 | 0.9190 | 0.9175 | 0.9113 |
| DistilBERT | 0.9445 | 0.9437 | 0.9419 | 0.9394 | 0.9289 |
| DistilBERT + typo augmentation | 0.9444 | 0.9424 | 0.9422 | 0.9396 | 0.9334 |

## Robustness summary

| Model | Clean F1 | F1 at 30% | Retention |
|---|---:|---:|---:|
| Word TF-IDF + LR | 0.9208 | 0.9022 | 0.9798 |
| Char TF-IDF + LR | 0.9204 | 0.9113 | 0.9902 |
| DistilBERT | 0.9445 | 0.9289 | 0.9835 |
| DistilBERT + typo augmentation | 0.9444 | 0.9334 | 0.9883 |

## Error types at p=0.20

| Model | Delete | Swap | Keyboard | Duplicate |
|---|---:|---:|---:|---:|
| Word TF-IDF + LR | 0.9121 | 0.9121 | 0.9125 | 0.9127 |
| Char TF-IDF + LR | 0.9167 | 0.9169 | 0.9179 | 0.9196 |
| DistilBERT | 0.9376 | 0.9360 | 0.9355 | 0.9375 |
| DistilBERT + typo augmentation | 0.9398 | 0.9389 | 0.9413 | 0.9401 |

## Short observations

- Highest clean Macro-F1: DistilBERT (0.9445).
- Highest Macro-F1 at p=0.30: DistilBERT + typo augmentation (0.9334).
- Augmentation changed Macro-F1 at p=0.00 by -0.0001 (augmented minus baseline).
- Augmentation changed Macro-F1 at p=0.30 by +0.0044 (augmented minus baseline).
