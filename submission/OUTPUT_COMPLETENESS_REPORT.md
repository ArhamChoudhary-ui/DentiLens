# Track 4 output completeness

| Dataset | Required | Usable inputs and valid outputs | Missing input |
| --- | ---: | ---: | ---: |
| EPIC-KITCHENS-100 | 138 | 5 | 133 |
| FPVSum | 49 | 3 | 46 |
| Total | 187 | 8 | 179 |

All eight source files matched the organizer inventory and generated
locally checked JSON. There were no failed inferences or invalid outputs.
Three short clips exceed the 15% target because the selected interval
meets the four-second minimum. Organizer validation remains unavailable.

- [Manifest audit with all missing IDs](../results/final_manifest_audit.json)
- [Final per-video output validation](../reference/dentilens-final-output-validation.md)
- [Reproduction and batch commands](../track4/README.md)
