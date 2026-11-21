# Instrument classifier confidence audit

The audited checkpoint is the 4.4 experimental RGB MobileNetV3 Small head. The runner verifies its SHA-256 against the training receipt and uses the released encoder and class order. The order is mirror, explorer, non-instrument, unknown. Frozen features and the head were trained from 18 mirrors, 24 explorers, 28 gloved-hand negatives, and 11 ambiguous crops. All 81 reviewed crops came from the training split. The original validation and test recording IDs remained unchanged. The archive lacks patient IDs, and the 4.4 crop similarity flags remain unresolved.

The four-output head correctly preserves non-instrument and unknown as separate raw outcomes. The gate accepts only named classes above 0.95. It does not forward non-instrument or unknown as an accepted instrument name. It had no raw named predictions at 0.95 on either evaluation split.

On the 1,281 test crops, correct raw mirror predictions had a median confidence of 41.7%. Correct explorers had a median confidence of 55.0%. The highest wrong named predictions reached 63.2% for mirrors and 80.7% for explorers. This overlap is a ranking and class-separation problem in addition to a calibration problem. The native confusion matrix includes 36 mirrors ranked explorer first, and 7 explorers ranked mirror first. The model also ranked an instrument name first on 95 of 1,082 non-instrument examples. Unknown and non-instrument need clean, diverse negatives beyond the original glove-heavy training subset.

The audit fits one temperature by validation negative log-likelihood. The fitted temperature is 0.61364. Validation was also used historically for checkpoint selection, so it is a calibration split, not independent confirmation. Scalar scaling keeps the class order and top-one predictions unchanged. Per-example diagnostics record logits, the top three, the margin, correctness, and the fixed acceptance grid for raw and scaled outputs. The grid measures behavior without choosing a gate from the locked test set. No class-specific calibrator was fit given the limited source groups and weak named precision.

| Split | Raw ECE | Scaled ECE | Raw Brier | Scaled Brier | Raw NLL | Scaled NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.1512 | 0.0231 | 0.4029 | 0.3697 | 0.7705 | 0.7091 |
| Locked test | 0.1885 | 0.0644 | 0.3550 | 0.3072 | 0.7050 | 0.6249 |

ECE uses ten bins and Brier is the sum of squared error across native classes. Native test top-one accuracy stays 80.17% and is dominated by negatives. The earlier grouped metric of 82.59% merges background and unknown and answers a different question. At the research gate, the scaled validation head accepted one explorer and it was wrong. On the locked test it accepted three correct explorers out of 59 and no mirrors. This covers only 0.23% of all test crops. Neither class demonstrated reliable 95% precision. The repeated benchmark is now familiar and needs a fresh, independently annotated set before making generalization claims.
