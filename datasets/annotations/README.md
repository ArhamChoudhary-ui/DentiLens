# Reviewed video annotations

Use a CVAT video task at original resolution and with original zero-based
frame indices. Label separate physical tools as separate tracks. Supported
labels are the canonical class IDs. Set CVAT outside for invisible or
out-of-frame tools and use the optional `visibility` attribute from the
schema to distinguish those two states. Use `ambiguous` when a box is
visible but tool identity is unclear. The importer interpolates visible
boxes between adjacent visible keyframes. Review that interpolation on
every scheduled sample.

A separate review JSON must include `review_state: verified`, `reviewer`,
`source_sha256`, `annotation_sha256`, and `frames`. The frame array must
be the complete scheduled list from extraction, including verified empty
frames. Hash the original CVAT XML after review. Unreviewed frames are not
negatives. Keep independent disagreement notes when a second reviewer
is available. Automatic proposals are never ground truth.
