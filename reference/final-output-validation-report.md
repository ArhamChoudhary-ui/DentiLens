# Track 4 output validation

The supplied manifest contains 187 unique IDs. It lists 138
EPIC-KITCHENS-100 videos and 49 FPVSum videos. The runner found eight
complete, matching source files and produced locally valid JSON for all
eight. Five results were added in this sprint and the three earlier
results were preserved byte for byte.

| Dataset | Expected | Available and valid | Missing inputs | Failed inference |
| --- | ---: | ---: | ---: | ---: |
| EPIC-KITCHENS-100 | 138 | 5 | 133 | 0 |
| FPVSum | 49 | 3 | 46 | 0 |
| **Total** | **187** | **8** | **179** | **0** |

The separate local checker found zero errors. The batch audit found no
duplicate input IDs, duplicate output IDs, corrupt JSON or unexpected JSON
files. It records the expected source and output filenames for every row
and the wall time for each new inference. The earlier three runs have CPU
analysis times in the [technical report](dentilens-track4-final-report.md).

| Video | Source length | Action intervals | Highlight intervals | Retained |
| --- | ---: | ---: | ---: | ---: |
| P02_13 | 29.51 s | 43 | 1 | 13.55% |
| P03_26 | 11.13 s | 15 | 1 | 35.94% |
| P26_30 | 15.28 s | 22 | 1 | 26.18% |
| P28_23 | 25.16 s | 37 | 1 | 15.90% |
| P32_07 | 28.98 s | 40 | 1 | 13.80% |
| RockClimbing_oVnAVcbMoSM | 83.50 s | 0 | 3 | 14.37% |
| Skate_el49s-RlIa4 | 37.24 s | 0 | 1 | 10.74% |
| Surfing_7bzMGVKaiCQ | 68.87 s | 0 | 2 | 11.62% |

The three warnings are for P03_26, P26_30 and P28_23. For each, 15% of
the clip is under four seconds. Each output keeps a single four-second
highlight to meet the guide’s minimum and incurs the documented budget
penalty. The official validator is still unavailable. Exact parity and the
organizer’s handling of short clips remain unverified.

The written student guide requires six top-level fields, scored verb–noun
records using a `segment` array and highlights containing only a
`segment` array. The local output follows that guide. The FPV example
uses `skiing`, while the supported vocabulary uses `ski`. Predictions
use `ski` and preserve the exact manifest video ID.

The [machine-readable manifest audit](../results/final_manifest_audit.json)
serves as the missing-input report.
