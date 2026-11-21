# Track 4 final output validation

The student-kit manifest has **187 unique IDs**, split into 138
EPIC-KITCHENS-100 and 49 FPVSum. The release audit found eight available
source clips. All eight outputs parse, match their source ID and expected
filename, use supported labels and valid intervals, and pass the local
contract checker. The audit found no failed inference, invalid outputs,
duplicate IDs or unsupported output labels. The remaining 179 entries
have no matching source in the checked workspace paths.

| Video | Source | Selected | Retained | Four-second minimum and target |
| --- | ---: | ---: | ---: | --- |
| P02_13 | 29.51 s | 4.00 s | 13.55% | Minimum met, within target |
| P03_26 | 11.13 s | 4.00 s | 35.94% | Minimum met, budget warning |
| P26_30 | 15.28 s | 4.00 s | 26.18% | Minimum met, budget warning |
| P28_23 | 25.16 s | 4.00 s | 15.90% | Minimum met, budget warning |
| P32_07 | 28.98 s | 4.00 s | 13.80% | Minimum met, within target |
| RockClimbing_oVnAVcbMoSM | 83.50 s | 12.00 s | 14.37% | Minimum met, within target |
| Skate_el49s-RlIa4 | 37.24 s | 4.00 s | 10.74% | Minimum met, within target |
| Surfing_7bzMGVKaiCQ | 68.87 s | 8.00 s | 11.62% | Minimum met, within target |

The [student guide](../track4/reference/student_package/TEAM_README.md)
requires at least one four-second highlight. It describes an over-15% check
as a warning and applies a soft length penalty in the metric. It does not
document a scoring exemption for short clips. On the three conflicts, the
selector keeps the four-second minimum and the budget warning. The
organizer’s actual validator remains unavailable.

The six-field submission schema includes finite scores and supported
verb–noun classes. Output timestamps are chronological where required,
bounded and rounded to two decimals. Highlight objects contain only
`segment`. The separate review format retains finite importance in
`[0, 1]`. FPVSum permits an empty action array. The written guide and
supported class file take precedence over the illustrative `skiing`
label, which conflicts with `ski` in the allowed vocabulary.

- [Machine-readable manifest audit](../results/final_manifest_audit.json)
- [Genuine EPIC and FPV sample JSON](../track4/reference/samples)
