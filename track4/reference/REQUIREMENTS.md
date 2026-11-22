# Track 4 requirements and student-package audit

The supplied student guide, vocabulary and manifest resolve the schema details
omitted from the original slides. The five received files are preserved unchanged
in reference/student_package/.

## Confirmed contract

- Each <video_id>.json requires team_id, video_id, duration_sec, label, actions
  and highlights. Labels contain name and score. Actions contain segment, verb,
  noun and score. Highlights contain segment. Importance stays in review data.
- The manifest has 187 unique IDs, comprising 138 epic_kitchens_100 entries and
  49 fpvsum entries. Duration ranges from 11.13 to 1968.6 seconds. There are
  74 videos outside the slides' typical 120–600 second range.
- Declared duration must match the manifest within 0.5 seconds. Segment ends
  allow a 0.5-second tolerance. Generated segments stay inside decoded footage.
- The dictionary contains 16 labels, 98 verbs and 301 nouns. Exact class spelling
  is checked. Scores must be finite numbers in [0, 1]. Actions are limited to 2000.
- Highlights are chronological and non-overlapping. At least one must last four
  seconds. The guide warns about shorter highlights and budget excess. Generated
  highlights all last at least four seconds.
- The four-second minimum conflicts with a strict 15% budget for P03_26 (11.13s),
  P26_30 (15.28s), and P28_23 (25.16s). The implementation selects one four-second
  clip and records an over-budget warning, following the guide's soft-penalty rule.
- Action coverage carries 40 points, action mAP 25, FPVSum highlight F1 25 and
  FPVSum category accuracy 10. EPIC uses the guide-prescribed cooking category.

## Package inconsistencies and remaining inputs

- validate.py is described in the guide but absent from the received folder.
  scripts/check_submission.py provides local checks from the written guide.
  Equivalence with the organizer's script awaits receipt of that script.
- Neither example ID, P02_109 or skiing_03, appears in the manifest. They remain
  format references. The skiing example uses unsupported label skiing, whereas
  the dictionary specifies ski. The local checker reports these inconsistencies.
- No raw videos, model weights or ground-truth annotations were included. The
  guide provides dataset download instructions and estimates roughly 184 GB
  for EPIC validation media.
- A registered team identifier must be set before submission export. Retain it
  in team_id.txt so the two-argument command works in an isolated environment.

## Submission rules retained from the slides

Preserve run.sh, code/, weights/, requirements.txt, outputs/ and README.md in the
real TEAM_ID folder. Generate all 187 outputs through the submitted system. The
student guide describes a hidden scored subset and reproducibility spot checks.
The slides also require a reproducible GitHub repository, deck and demo. The
stricter three-minute demo limit is used for planning. The November 18 timeline
conflicts with an October 28 date on one track slide.

## Model status

The verified encoder is MCG-NJU/videomae-base at revision
dc740ceda42fce44faed2ea03c6d447db72f6af9. Its model card describes self-supervised
Kinetics-400 pretraining and lists CC-BY-NC-4.0. Frozen features feed separately
trained linear verb, noun and category heads. The optional custom path requires trained heads. Training and checkpoint loading check vocabulary compatibility.

Previously investigated model sources are
https://huggingface.co/MCG-NJU/videomae-base and
https://github.com/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM.
