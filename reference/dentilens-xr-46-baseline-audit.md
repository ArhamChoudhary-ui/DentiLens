# DentiLens XR 4.6 baseline audit

The application stores uploads and normalized source video locally under
the examination workspace. Instrument scans and identity suggestions live
in separate examination sidecars. Reviewer corrections stay in SQLite and
are excluded from automatic comparisons. The 4.5 shadow route reads the
saved production output, runs against the same detector proposals, and
writes its own comparison sidecar. Application uploads have no ML-use
clearance implication. The new acquisition registry is a separate,
permission-gated workflow.

DentalGuard 4.3 manifests bind the archive and preserve its original
recording groups. The 4.4/4.5 test set has been scored repeatedly. The
temperature fitter uses validation logits only, and the archived CLI
rejects alternate-resolution experiments on test. The tests and code
audit found no locked-set fitting. The 95% threshold was fixed before
scoring. The old cohort can reproduce a known benchmark, while promotion
requires newly acquired, untouched recordings. No such recordings have
been registered.

Recomputing native-class test ECE from the saved per-crop top probability
and correctness produced **0.1885138122** raw and **0.0643612906** calibrated
on 1,281 crops. ECE is the weighted mean of absolute confidence-versus-
accuracy gaps in ten equal-width bins, with confidence 1 assigned to the
last bin. It is a calibration statistic. Native top-one accuracy remained
80.17%. At the fixed gate, three accepted test explorers were correct.
The sample-level Wilson 95% interval for 3/3 is roughly **44%–100%**;
the crops are clustered by recording, so this illustration does not
establish recording-level precision. Validation had one incorrect
accepted explorer. No mirrors were accepted.

The locked detector baseline remains **26/526** for the released 512 px
detector and **0/526** for the still-photo model at its own cutoff. The
validation-only rerun reproduced 30 matches at 512 and 36 at 768. All
three resolutions use normalized source coordinates, the same held-out
validation rows, checkpoint, score gate and rank floor. Matching is
one-to-one, with no requirement for a detector class match because the
video detector has one generic class. The error ledger distinguishes
overlap with an already matched tool, annotated context, partial overlap,
and low-overlap unverified proposals. Missing annotations and temporal
coverage prevent video-wide clinical false-positive or duration claims.

No checkpoint or coordinate bug was confirmed. Weak localization,
background proposals, incomplete labels and limited source diversity
remain the dominant observed limitations. Production resolution, weights,
thresholds and tracker were preserved.
