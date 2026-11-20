# Model files

Run `python scripts/download_model.py` to retrieve the published AlphaDent
checkpoint. The downloader verifies its SHA-256 before installing it here.

The source URL and measured hash are retained in `reference/model-source.json`.
The publisher's accompanying license is retained in `reference/AlphaDent-LICENSE`.
Ultralytics is a separate dependency with its own licensing terms.

For tooth-region masks, run `python scripts/download_grounded_models.py` after
installing `requirements-grounded.txt`. This installs pinned detector and SAM 2
weights under `models/grounded/` and verifies their SHA-256 values.

The optical research checkpoint has been downloaded locally. Its contour stage
failed on the tested braces footage, so the active tooth-region engine uses
Grounding DINO with SAM 2.1.
