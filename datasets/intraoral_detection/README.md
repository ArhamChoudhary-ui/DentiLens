# Intraoral instrument detection

Build the source-linked `frames.json` manifest with `scripts/build_dataset.py`.
Each row identifies an archived image, source recording, frame number, split,
dimensions, and annotation version via the dataset metadata. The original OBB
and normalized enclosing AABB are retained. A non-instrument region records a
negative annotation within a frame. Its presence alone does not certify a
negative full frame.

For new CVAT jobs, use one task per recording with its stable recording ID
and original frame rate. Draw a tight box around each visible tool and mark
unidentifiable tools unknown. Assign non-instrument to reviewed background
regions. Mark tool-free frames as verified negative only after inspecting the
whole frame. Record visible/partially visible and occlusion as attributes.
Preserve original frame index, timestamp, parent source checksum, annotation
version, and correction history when converting to the manifest format. The
archive importer uses null for timestamps, visibility, and occlusion that the
source lacks. CVAT export conversion for new recordings remains separate from
the DentalGuard archive importer.
