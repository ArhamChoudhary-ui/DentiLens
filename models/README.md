# Model files

Run `python scripts/download_model.py` to retrieve the published AlphaDent
checkpoint. The downloader verifies its SHA-256 before installing it here.

The source URL and measured hash are retained in `reference/model-source.json`.
The publisher's accompanying license is retained in `reference/AlphaDent-LICENSE`.
Ultralytics is a separate dependency with its own licensing terms.

The optical tooth-segmentation model remains under investigation. Its checkpoint
download has been interrupted by a server certificate verification error.
