# Instrument recognition recovery

## Where the names were lost

The 4.1 baseline reproduced on all 18 saved proposals. The general head ranked unknown first on 17. Its only other top prediction was dental handpiece at 96.7% on a gloved forearm at 1.40 s in clip 4489972. That scored below the existing 98% handpiece threshold. The mirror specialist ranked unknown first on all 18. The context head passed its frame-level gloved-hand gate on the forearm crop. There were no accepted instrument frame predictions to discard in tracking or UI rendering.

Visual review of the 18 crops found five chair/background regions, one gloved forearm, and twelve glove-dominated scenes that may contain a partial tool. The eleven proposals on 14195516 repeatedly framed the hand alongside a narrow metal shaft. There was insufficient visible tip detail to assign a specific tool to those crops. A mouth mirror is visible in the opening of 4489972 before the detector starts proposing boxes. This is evidence of a proposal-quality problem as well as a classifier-domain problem.

Three manually boxed mirror frames at 0.00, 0.40, and 0.80 s each had zero detector proposals. The specialist ranked mirror first at 0.00 s with 90.4%, below the 95% mirror cutoff. The general head passed its frame gate with an incorrect forceps prediction (81.9%) at that timestamp. Neither head produced a correct accepted mirror name on these three boxes. These repeated frames from one clip isolate classifier error even when localization is supplied. They are a diagnostic probe, not an independent benchmark.

## Checkpoint integrity

The audit loaded the installed general identity head, mirror specialist, and DentalGuard context head through the runtime loader. Their state dictionaries loaded against the declared MobileNetV3 Small architectures after SHA-256 verification. The ordered classes, aliases, paths, modification times, and thresholds are in the audit JSON. The training recipe and audit use the same padded crop encoder and MobileNet transform. The general head uses grayscale expanded by normalization to three channels. The specialist and context heads use RGB. The encoder pads the longest dimension to a 96-pixel square before the shared 128-pixel resize and ImageNet channel normalization.

The explorer exists in both identity heads, but neither currently enables it at the release operating point. That is a separate limitation from the 18-proposal failure. Explorer never ranked first on those proposals.

## Controlled crop evaluation

Accuracy and macro-F1 below use raw top-one predictions over the configured class list. Coverage and per-class precision/recall at the deployed gates are recorded separately in the audit JSON. Photo validation was used for checkpoint selection and gate setting. The photo test set is reported separately. Both use the original archived split with the training recipe's filename and byte-hash deduplication.

| Head and split | Crops | Top one | Top three | Macro-F1 | Rejected at frame gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| General, photo validation | 1,356 | 75.0% | 98.2% | 0.744 | 1,221 |
| General, photo test | 700 | 74.1% | 98.1% | 0.516 | 609 |
| Specialist, photo validation | 1,356 | 74.0% | 100% | 0.535 | 1,345 |
| Specialist, photo test | 700 | 81.0% | 100% | 0.569 | 690 |
| General, DentalGuard test | 253 | 39.1% | 73.9% | 0.074 | 246 |
| Specialist, DentalGuard test | 253 | 45.5% | 100% | 0.290 | 253 |
| Context, DentalGuard test | 419 | 82.8% | 98.1% | 0.831 | 147 |

The specialist has three classes, so its top-three score is automatic. On the same four DentalGuard test source groups, the specialist ranked mirror first on 2/82 mirror crops and explorer first on 8/59 explorer crops. The general head ranked mirror first on 4/82 and explorer first on 0/59. At the deployed frame gates, the general head accepted 7/253 names. It accepted two mirror predictions, of which one was correct (50% precision, 1.2% recall), and accepted none for explorer. It named 2/112 negative crops. The specialist accepted zero names. Raw top-one expected calibration error with ten fixed bins was 0.478 for the general head and 0.420 for the specialist. These are supplied-box classification results from source-separated frames. They do not measure detector recall or streaming performance. Precision, recall, macro-F1, calibration, and false positives for the two complete new clips require full ground truth.

## Dataset audit and next experiment

The V3 archive contains 5,213 exported 640 × 640 grayscale images with generated rotation and flip variants. Its label filenames reduce to 2,153 source-name groups. SHA-256 found 5,198 byte-distinct files and 15 exact duplicates. The original per-class box counts and normalized box size distribution are in the JSON. The median box occupies about 46% of image width and 42% of height. The two mirror labels account for 578 boxes and Explorer No.5 for 497 in the grouped census. The general head's grouped photo training split contains 434 mirror crops, 382 explorer crops, and 4,486 crops mapped to unknown. Patient/source IDs and reliable scene-type labels are absent from this archive. The named DentalGuard split provides verified source IDs from filenames and annotated hard negatives. Identity beyond those source IDs is unverified. Both local archive notices declare CC BY 4.0.

The results justify a source-separated intraoral training experiment after the training examples and labels are assembled. Start with a detector benchmark to measure recall for visible mirrors and identify false boxes around gloves and the chair. Compare classification on matched detector crops and manually boxed tools to decide which stage needs the next change. Include occlusion, glare, distant instruments, and confirmed glove-only, tooth-only, and brace-only negatives. Keep the two audit clips evaluation-only. No model was retrained or promoted, and thresholds and mirror tracking were kept in place.

## Reproduce

```bash
python scripts/audit_instrument_recovery.py \
  /path/to/instrument-4.1-audit.json /path/to/V3.zip /path/to/clips /path/to/output \
  --dentalguard /path/to/DentalGuard.zip \
  --mirror-probes reference/instrument-recovery-mirror-probes.json
```

Generate the 4.1 audit JSON first with `scripts/audit_instrument_crops.py`. The new audit writes a browsable HTML report, contact sheet, exact encoded crop images, and a structured JSON report. It uses the same classifier and tracking paths for final status, and explicitly records each head's top-three probabilities, logits, margin, and frame gate separately. Keep raw media and generated image artifacts local.

## Raw top-three predictions on the 18 proposals

Scores are probabilities before frame and track acceptance. The context head predicts scene content.

| Clip / time | General | Mirror specialist | Context | Final reason |
| --- | --- | --- | --- | --- |
| 4489972 / 1.20s | unknown 99.68%; dental_explorer 0.15%; dental_forceps 0.08% | unknown 99.94%; dental_explorer 0.05%; dental_mirror 0.01% | unknown 99.61%; mouth_open 0.20%; hand_gloved 0.19% | class_unknown |
| 4489972 / 1.40s | dental_handpiece 96.74%; unknown 2.46%; orthodontic_pliers 0.80% | unknown 98.33%; dental_mirror 1.45%; dental_explorer 0.22% | hand_gloved 98.44%; hand_ungloved 1.43%; unknown 0.13% | below_class_threshold |
| 4489972 / 8.80s | unknown 70.27%; orthodontic_pliers 23.44%; wire_cutter 3.13% | unknown 96.86%; dental_mirror 1.80%; dental_explorer 1.34% | unknown 100.00%; mouth_open 0.00%; hand_ungloved 0.00% | class_unknown |
| 4489972 / 9.20s | unknown 59.00%; orthodontic_pliers 37.73%; dental_forceps 2.04% | unknown 93.05%; dental_explorer 3.68%; dental_mirror 3.27% | unknown 100.00%; mouth_open 0.00%; hand_ungloved 0.00% | class_unknown |
| 4489972 / 9.80s | unknown 60.35%; dental_forceps 37.87%; orthodontic_pliers 1.69% | unknown 67.87%; dental_explorer 30.95%; dental_mirror 1.18% | unknown 100.00%; mouth_open 0.00%; hand_ungloved 0.00% | class_unknown |
| 4489972 / 10.00s | unknown 70.15%; dental_forceps 27.69%; orthodontic_pliers 2.06% | unknown 72.91%; dental_explorer 25.98%; dental_mirror 1.11% | unknown 100.00%; mouth_open 0.00%; hand_ungloved 0.00% | class_unknown |
| 4489972 / 10.20s | unknown 74.27%; dental_forceps 24.05%; orthodontic_pliers 1.62% | unknown 69.27%; dental_explorer 29.29%; dental_mirror 1.44% | unknown 100.00%; mouth_open 0.00%; hand_ungloved 0.00% | class_unknown |
| 14195516 / 0.00s | unknown 73.31%; dental_handpiece 25.37%; wire_cutter 0.51% | unknown 99.88%; dental_mirror 0.08%; dental_explorer 0.04% | unknown 99.98%; hand_gloved 0.02%; mouth_open 0.00% | class_unknown |
| 14195516 / 0.20s | unknown 83.35%; dental_handpiece 14.10%; bracket_holder 0.91% | unknown 98.59%; dental_explorer 0.86%; dental_mirror 0.55% | unknown 100.00%; hand_gloved 0.00%; mouth_open 0.00% | class_unknown |
| 14195516 / 0.40s | unknown 93.77%; dental_handpiece 5.26%; bracket_holder 0.36% | unknown 99.89%; dental_explorer 0.08%; dental_mirror 0.03% | unknown 100.00%; hand_gloved 0.00%; mouth_open 0.00% | class_unknown |
| 14195516 / 1.60s | unknown 99.78%; orthodontic_pliers 0.12%; dental_handpiece 0.06% | unknown 99.11%; dental_mirror 0.86%; dental_explorer 0.03% | unknown 99.99%; hand_gloved 0.01%; mouth_open 0.00% | class_unknown |
| 14195516 / 1.80s | unknown 99.92%; orthodontic_pliers 0.04%; dental_handpiece 0.03% | unknown 99.96%; dental_mirror 0.04%; dental_explorer 0.00% | unknown 99.97%; hand_gloved 0.03%; mouth_open 0.00% | class_unknown |
| 14195516 / 2.00s | unknown 98.84%; dental_handpiece 1.07%; orthodontic_pliers 0.04% | unknown 99.61%; dental_mirror 0.37%; dental_explorer 0.02% | unknown 99.99%; hand_gloved 0.01%; mouth_open 0.00% | class_unknown |
| 14195516 / 3.40s | unknown 97.46%; dental_handpiece 2.54%; orthodontic_pliers 0.00% | unknown 99.98%; dental_mirror 0.02%; dental_explorer 0.00% | unknown 99.92%; hand_gloved 0.08%; mouth_open 0.00% | class_unknown |
| 14195516 / 3.60s | unknown 96.42%; dental_handpiece 3.58%; dental_explorer 0.00% | unknown 99.98%; dental_mirror 0.02%; dental_explorer 0.00% | unknown 99.19%; hand_gloved 0.81%; mouth_open 0.00% | class_unknown |
| 14195516 / 3.80s | unknown 95.00%; dental_handpiece 4.96%; orthodontic_pliers 0.03% | unknown 99.98%; dental_mirror 0.02%; dental_explorer 0.00% | unknown 99.97%; hand_gloved 0.03%; mouth_open 0.00% | class_unknown |
| 14195516 / 4.00s | unknown 96.89%; dental_handpiece 3.05%; dental_explorer 0.06% | unknown 99.94%; dental_mirror 0.05%; dental_explorer 0.00% | unknown 99.94%; hand_gloved 0.06%; mouth_open 0.00% | class_unknown |
| 14195516 / 5.00s | unknown 94.51%; dental_handpiece 5.34%; orthodontic_pliers 0.08% | unknown 99.98%; dental_mirror 0.02%; dental_explorer 0.00% | unknown 99.75%; hand_gloved 0.25%; mouth_open 0.00% | class_unknown |
