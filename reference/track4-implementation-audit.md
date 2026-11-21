# Track 4 implementation audit

The organizer PDF and supplied student kit refer to action-aware
egocentric video summarization. The guide and examples use the detailed
six-field contract, which supersedes the shorthand diagram in the
slides. The project retains the dental workflow and integrates the
earlier standalone Track 4 implementation inside `track4/`.

| Requirement | Existing capability | Work and remaining gap | Priority |
| --- | --- | --- | --- |
| Input and source timing | Dental video and earlier Track 4 MP4 readers | Use the prior CFR reader. Match original ID and duration. The TSN stream decodes one pass. | P0 |
| Organizer classes and coverage | Five organizer files in both kits | Verified 187 IDs (138 EPIC and 49 FPVSum), with 16 labels, 98 verbs and 301 nouns. Neither sample ID appears in the current manifest. | P0 |
| Official validator | Guide describes `validate.py` | File absent in the organizer ZIP, extracted copy and existing tracked kit. Preserve local checker identity. | P0 |
| Actual recognition | VideoMAE encoder and optional custom heads | Integrate verified EPIC TSN weights and Kinetics R3D-18 transfer. Dental weights are separate. No contact validation or learned boundary head. | P0 |
| Temporal windows and highlights | Selector, temporal NMS, source MP4 compiler | Retain earlier code. Verify short-video contract conflict and record warnings. | P0 |
| Full metrics | Organizer protocol described at a high level | Add separate original-public-label diagnostics. Complete media and organizer parity are pending. | P0 |
| Offline package | Prior CLI, shell wrapper and Docker sketch | Verified checkpoint assets and output gate. Container configuration provided. Docker build not exercised. | P0 |
| Application UI | Existing dental review interface | Dedicated CLI and local evidence first. In-app Action Intelligence has not been added while P0 remains incomplete. | P2 |
| Headset and XR use | Simulated XR in the dental workspace | Track 4 is a local prototype. No device deployment measurement. | P2 |

The organizer sample uses `skiing`, while its supported class is `ski`.
Manifest IDs remain unchanged. The 4-second minimum conflicts with
the 15% budget on three manifest videos shorter than 26.67 s.
Both constraints are documented in the implementation. The final
submission deadline needs organizer confirmation because the supplied
slides contain conflicting dates and the team reported a later date.

Pretrained sources: [EPIC baseline](https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM),
[EPIC class and training annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations),
[R3D-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.r3d_18.html),
[FPVSum public data](https://github.com/azuxmioy/fpvsum).
