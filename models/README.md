# Model files

Run `python scripts/download_model.py` to retrieve the published AlphaDent
checkpoint. The downloader verifies its SHA-256 before installing it here.

The source URL and measured hash are retained in `reference/model-source.json`.
The publisher's accompanying license is retained in `reference/AlphaDent-LICENSE`.
Ultralytics is a separate dependency with its own licensing terms.

For tooth-region masks, run `python scripts/download_grounded_models.py` after
installing `requirements-grounded.txt`. This installs pinned detector and SAM 2
weights under `models/grounded/` and verifies their SHA-256 values.

The locally trained intraoral detector installs as `tooth-detector.pt` with a
matching `tooth-detector.json` provenance file. Run
`python scripts/install_tooth_detector.py /path/to/completed-training-run`.
The `tooth-sam2` engine verifies the detector hash before inference and only
needs the SAM component, installed with
`python scripts/download_grounded_models.py --only sam2`.

Dataset attribution, the one-class conversion policy and evaluation scope are
recorded under `reference/`. Ultralytics tooling and pretrained model terms are
available at https://www.ultralytics.com/license. Dataset metadata declares
CC BY 4.0. Local weights, source images and footage remain excluded from Git.

The earlier optical research checkpoint's contour stage failed on the tested
braces footage. It is excluded from the active engines. The text-guided baseline
continues to use Grounding DINO with SAM 2.1.

| App mode | Local files | Training data |
| --- | --- | --- |
| Tooth type names | `tooth-types.pt`, `tooth-types.json` | yusma intraoral v2, eight tooth types |
| Tooth numbers | `tooth-fdi.pt`, `tooth-fdi.json` | DentalMate6v FDI v1, 32 permanent positions |
| Possible cavity flags | `cavity-review.pt`, `cavity-review.json` | CAV-TEE Research Team, cavity and normal box annotations |

The installer uses the recorded preparation task to select the destination.
The tooth type mode also uses the locally installed SAM 2.1 model.
