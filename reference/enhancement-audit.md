# DentiLens XR enhancement audit

The existing React workspace and FastAPI service share SQLite examination and
review records. Uploads, source video, rendered media and frame records are
local. The worker normalizes video for playback, samples detections and renders
every normalized frame. ByteTrack links geometry and bidirectional optical flow
supplies intermediate measurements. The three YOLO checkpoints suggest tooth
type, permanent FDI number and possible cavities. Grounding DINO selects a crop.

Baseline regression suite passed all 18 tests. The recorded braces run decoded
675 frames at 25 fps. It exported 675 frames and rendered overlays on 291.
The remaining 384 frames have empty annotation records. The old frame log has
no detector rejection reasons or ground-truth visibility labels, so it cannot
establish how many were missed. The full-video renderer verifies the encoded
frame count before publishing the video.

The grounder can reject the entire frame before either tooth identity model
runs. The tracker refreshes early when all boxes disappear, while partial
losses can wait for the next scheduled detection. Evidence selection favors
identity and detector scores. It has no crop quality measure. The existing
review stores the original AI finding separately but provides only a free-text
label. The reviewed reel reflects a saved correction. The full MP4 preserves
the original AI predictions.

Model provenance and dataset preparation are recorded under `models/` and
`reference/`. The eight-class type and 32-class FDI models have photo test
results. Existing video clips have no independent frame-level tooth or
eligibility labels. The installed SAM model supports the separate region
segmentation workflow. The current full-video identity pipeline uses boxes.

## Validation after enhancement

All five full recordings in the validation set exported at their expected
normalized frame counts. The braces comparison used Detailed for both runs.
Baseline overall coverage was 291 of 675 frames (43.1%). The enhanced pass
produced overlays on 398 of 675 frames (59.0%), including 302 frames with
individual tooth boxes. FDI suggestions appeared on 68 frames. Both runs
used 57 detection passes. The enhanced run took 135.96 seconds with a warm
model on Apple Silicon CPU. MPS was unavailable in the execution session.

The 277 blank frames break down by pipeline observation into 253 intervals
between detection passes, 19 passes with no accepted detection, four below
the region search threshold and one loss of tracking. Those categories are
mutually exclusive. They do not establish whether a tooth was visible or
why the model missed it. Two additional recordings produced zero overlays
in Fast mode. The video data do not establish accuracy, F1, calibration,
clinical validation or eligible-frame coverage. The coverage script accepts
independent visibility annotations if they become available.

The full-video runner still normalizes to a 1280-pixel long edge at up to 30
fps. The corrected export reuses that resolution and timebase. Models are
unchanged. The new coverage reflects crop recovery, spatial screening and
tracking changes. Individual tooth masks, calibration, dataset retraining
and physical XR validation require separate evidence or equipment.
