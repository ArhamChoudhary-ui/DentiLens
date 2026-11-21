# Preparing a complete recording benchmark

Register one consented source recording per stable recording ID. Record its
SHA-256, exact frame rate/time base, source dimensions, annotation version,
rights evidence, and patient group where available. Keep all annotations,
extracted frames and derived crops from the examination in that source group.

Create one CVAT video task for each recording. Draw a separate object track
for each instrument and retain its track ID through partial occlusion. Use
the canonical instrument class and an explicit unknown label for ambiguous
instruments. Mark occlusion and outside intervals. Review entry and exit
timestamps and simultaneous instruments. Record independently reviewed
tool-free intervals explicitly. Export CVAT XML plus the recording metadata.
Do not share raw patient media in the repository.

A future evaluator should map predictions and reviewed boxes to the same
source frame index, match per frame at a fixed IoU threshold, and keep the
ground-truth and prediction track IDs. Measure visible annotated duration
before summing misses. Count fragmentation only across intervals with known
visibility and count identity switches only when the matched prediction ID
changes. Record time-to-first-correct-recognition from each reviewed entry.
Report per-recording results and aggregate with source grouping. Include
absence intervals for false-track measurement. The current frame archive
has neither complete recordings nor ground-truth object trajectories.
