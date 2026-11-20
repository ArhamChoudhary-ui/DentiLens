# DentiLens XR 3.5 delivery audit

Baseline commit `7cc446c` has 21 passing tests. The application uses a local
FastAPI and SQLite backend, full-duration normalized video and frame JSONL,
and a React review workspace. Tooth detection, editing, patient bookmarks,
XR preview and original exports retain their existing entry points. A new
SQLite table stores instrument decisions. Old examinations can be scanned
directly with the new instrument model.

## Instrument data and training

The user-provided Dental Instruments v30 export declares CC BY 4.0. Its
3605 images use normalized OBB labels for 14 classes. The preparation
pipeline validates decoded images and quadrilaterals, keeps one version
per filename-derived source group and excludes 19 cross-split groups.
The resulting training, validation and test counts are 1053, 166 and 152.
Patient identifiers are unavailable. Low-distance dHash pairs are flagged
for inspection. The photo split cannot establish patient-level independence.

The first CPU training stage used the Ultralytics YOLOv8n OBB initialization
for 3 epochs, followed by 8 more epochs initialized from its best validation
checkpoint. The final held-out photo metrics were 0.389 mAP50, 0.226
mAP50–95, 0.300 precision and 0.602 recall. The initial and refinement
runs took 122.97 and 294.94 seconds including test evaluation.
The tool runtime reported MPS unavailable. Runtime instrument inference
uses CPU and samples approximately five frames per second.

The measurement record contains per-class results for both stages.
The separate video report counts suggestions emitted on five local clips.
Those clips do not have instrument ground truth. Scores and classifications
remain model suggestions. Photo metrics cannot establish instrument accuracy
on close-up examination footage. No spatial overlap or co-occurrence is
presented as evidence of a procedure.

## Anatomy and export

The viewer loads the unmodified OrganUI GLB. The loader test checked all
33 named meshes and all 28 permanent-tooth mappings. The selected position
links back to the local chart and history. Missing third molars retain their
chart positions. The model is educational anatomy. It is an atlas asset,
and no patient-specific geometry is generated.

The exported braces demo retained 675 frames, 25 fps and 27 seconds despite
adding optional instrument polygons. Instrument marks use their observation
frames. The software regression fixture checks a corrected instrument and
full-video export alongside all existing review routes. Anatomical data and
instrument checkpoints use separately documented sources and terms.

## Remaining scope

IOP-Compass acquisition, individual-tooth mask validation, orthodontic
instrument classes, temporal procedure recognition, audio transcription,
live capture, multi-visit comparison, and collaboration are outside this
delivery. Consult the dataset registry for remaining access and annotation
requirements. No changes were made to the separate Track 4 benchmark.
