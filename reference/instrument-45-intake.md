# Instrument dataset intake

Record one JSON object per reviewed crop. Include `id`, `source` (URL or
archive citation), `recording`, `filename`, `parent_frame` (stable original
frame index), `reviewer`, `dataset_version`, `permission_evidence`,
`label`, normalized `box`, `width`, `height`, `split`, `status`,
`identifiable`, `sha256`, and `dhash`. Use versioned consent or archive
records as provenance. Store confidential authorization records locally.

Record `permissions.extract`, `permissions.annotate`, `permissions.train`,
and `permissions.evaluate`. For identifiable or unverified footage, also
record explicit `permissions.identifiable_training`. Keep the identity
reviewer separate from the license record. Put unclear crops in
`status: quarantine` and reserve `status: verified` for a specifically
identified or explicitly negative crop. Complete a full-frame privacy
review separately before detector training. Follow the
[recording annotation protocol](instrument-44-recording-protocol.md)
for complete video ground truth.

Run `.venv/bin/python scripts/validate_instrument_intake.py /path/to/intake.json`.
It checks declared permissions, the locked groups, recording split isolation,
exact fingerprints, and perceptual similarity across new splits. The validator
does not independently prove rights or calculate hashes from raw media.
Keep different recordings, patients and typodont captures distinguishable.
No newly cleared recording was added in 4.5.
