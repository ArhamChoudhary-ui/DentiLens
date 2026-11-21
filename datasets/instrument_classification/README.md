# Instrument crop classification

Build the independent `crops.json` manifest with `scripts/build_dataset.py`.
Rows reference parental detection annotations and retain the original class
label and box. Crops are extracted on demand by the released checkpoints'
production preprocessing. An explicit non-instrument label is distinct from
unknown tool type and other named tools outside the core vocabulary. Padded
negative crops overlapping any annotated instrument are excluded. A negative
crop still depends on source annotation completeness.
