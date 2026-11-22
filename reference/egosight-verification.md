# EgoSight XR verification

Verified locally on 22 November 2026 using the Apple Silicon Mac CPU path. The browser calls FastAPI, which runs the independent Track 4 CLI in a worker. The server validates the result before it makes the original JSON available.

## Inference on organizer-listed videos

| File | Returned category | Timed actions | Highlights | Local validation |
| --- | --- | ---: | ---: | --- |
| `P02_13.mp4` | cooking (manifest category) | 43 | 1 | PASS |
| `RockClimbing_oVnAVcbMoSM.mp4` | rockclimbing | 0 | 3 | PASS |
| `Skate_el49s-RlIa4.mp4` | other | 0 | 1 | PASS |
| `Surfing_7bzMGVKaiCQ.mp4` | other | 0 | 2 | PASS |

The JSON download matched the runner output for all four clips. The FPVSum transfer model supplies activity categories and highlights. Its action list is empty. An upload of the climbing footage under an unlisted name also returned rockclimbing with three highlights and passed schema validation. This verified the unlisted-upload path with the same source footage.

## Browser, API and CLI

| Check | Result |
| --- | --- |
| Chromium upload, preview, processing state and model result | PASS |
| Timeline seek and highlight playback to the end of the selected interval | PASS |
| Open JSON, download and compare it with the runner file | PASS |
| Reopen saved history and open benchmark and model settings | PASS |
| Mobile overview at 390 pixels, desktop workflow at 1440 pixels | PASS |
| Dental module navigation and opening an existing examination | PASS |
| New dental model inference across every mode | NOT TESTED |
| FastAPI health, settings, errors, malformed upload cleanup and concurrent-run rejection | PASS |
| Pytest and Vite production build | PASS (102 tests) |
| Staging ZIP integrity and repeat inference from its launcher | PASS (byte-for-byte P02_13 JSON match) |

The production build reports a size warning for the existing dental anatomy bundle. The package uses the existing TSN RGB ResNet50 checkpoint `tsn_rgb.ckpt` and Kinetics-400 R3D-18 checkpoint `r3d_18-b3b3357e.pth`. This release integrates pretrained weights. It includes no new training or tuning. A new training run requires authorized labeled training clips and a separate source-level validation split. Keep the competition annotations reserved for evaluation.

## Samsung staging coverage

The staging ZIP contains 102 valid outputs out of 185 eligible IDs, including 55 EPIC-KITCHENS-100 and 47 FPVSum clips. The supplied manifest still lists all 187. The two approved unavailable scuba exclusions are recorded in [the organizer update](../submission/ORGANIZER_UPDATE.md). The 83 outstanding eligible videos and the official organizer score remain pending. The local checker verifies output shape, vocabulary and timing. Samsung’s weights are action coverage 40%, action mAP 25%, highlight F1 25% and label accuracy 10%.

The written highlight rules require a four-second segment and target at most 15% of source duration. For very short clips, the checker records a warning when those constraints conflict.
