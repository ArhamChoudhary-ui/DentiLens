# DentiLens XR 3.0 audit

Baseline is Git commit `0e7384d`. All 19 tests passed before changes.
The React and Vite app calls FastAPI. SQLite stores the examination, latest
reviews and review history. Per-examination directories hold normalized video,
MP4 exports, frame JSONL, and JPEG evidence. A single local worker runs
Grounding DINO and the installed YOLO tooth type, FDI and optional cavity
models. ByteTrack and checked optical flow continue observations between
detection passes. Existing rendering, patient view and XR preview are covered
by the earlier audit and tests.

The current chart links suggested FDI numbers to selected bookmarks. It lacks a
frame-level history or a searchable index of tooth tracks and reviewer notes.
The frame records already support those workflows, including for older
examinations. Their track IDs identify observations within one recording.
A frame-level FDI label is a suggestion even when the same number appears
on more than one track. Reviewer confirmation is carried separately.

First delivery uses the existing frame records and live review rows to build
tooth appearances, a visibility map and source-linked search. Measurements
of recording quality are added as frame metadata and summarized into
navigable cues. Existing records remain readable. Results distinguish model
scores, reviewer decisions and measurements. A suggested identity is never
used to infer disease or patient-level continuity.

General SAM weights support region prompts in the legacy workflow. The
current full-video model has measured box behavior and lacks validated
individual-tooth mask evaluation. Instrument localization in the legacy
workflow is experimental. Procedure activity recognition, audio transcription
and additional condition models require appropriate inference and validation
assets. The Track 4 organizer manifest, classes and examples are separate
from the dental application.

## Track 4 package audit

The supplied manifest lists 187 video IDs. Neither example ID is in the
manifest. The downloaded ZIP contains the guide, vocabulary, manifest and
two format examples. It does not contain the guide's `validate.py`. The
skiing example uses `skiing`; the vocabulary lists `ski`. The example checker
reports the discrepancy. The existing root `run.sh` launches the dental
inference pipeline. A separate benchmark runner needs suitable EPIC-KITCHENS
and FPVSum assets, action and video-label checkpoints, and the official
validator from the organizers before it can produce validated submissions.

Run the local check with `python scripts/check_track04.py --outputs /path/to/outputs`.
This reports schema violations and manifest coverage. The check deliberately
preserves the supplied example and class files.
