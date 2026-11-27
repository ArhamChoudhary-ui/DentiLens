# Samsung Track 4 organizer update

- The team may use its own validator for syntax and edge-case checks.
- The manifest lists 187 videos. Samsung excluded `Scuba_KD74Ywuy9ng`
  and `Scuba_Um4kqB8bQBg` from scoring, leaving 185 eligible videos.
- An action counts as captured when at least half its duration lies inside
  the union of the selected highlights. Predicted labels are scored
  separately through action mAP.
- The supported `ski` label takes precedence over illustrative examples.
- Samsung specified no hardware or runtime limits for re-runs and advised
  treating the FPVSum annotations as hidden test labels.

The converted [student guide](../track4/reference/student_package/TEAM_README.md)
and [example JSON](../track4/reference/student_package/skiing_03.json)
come from the updated attachments. The guide retains the original 187-row
instruction. The email supplies the two specific exceptions.
