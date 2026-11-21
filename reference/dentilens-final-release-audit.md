# DentiLens XR release audit

| Area | Current state | Verification evidence | Remaining work |
| --- | --- | --- | --- |
| Repository | Existing Track 4 branch and dental app retained | Started clean at `6833f2a` on `xr-4-video-intelligence`. Local checkpoint `checkpoint/track4-before-final-release` records the starting commit. | Use the published release commit for the form. |
| Manifest and input sources | 187 unique IDs, eight matched source videos | Checked the student-kit CSV and configured workspace data, datasets and runtime locations. The audit lists 179 missing source filenames. | Acquire missing sources or obtain organizer permission for a partial upload. |
| Outputs | Eight genuine JSON files passed local validation | Revalidated all outputs, retained all eight files and checked regenerated samples byte for byte. | Run the official validator when obtained. |
| FPVSum | Two genuine category misses retained | Re-ran skate and surfing, inspected raw logits, verified checkpoint and category mapping, and reproduced both JSON files byte for byte. | Evaluate a more suitable first-person classifier on new held-out clips. |
| Tests and offline inference | Pass | Existing 99 tests and two-argument inference from the extracted staging ZIP. | Run organizer checks with the official validator and intended team ID. |
| Validator | Organizer script absent | Git history, repo, supplied extracted kit and archive have no official `validate.py`. | Obtain the organizer copy. |
| PPTX and PDF | Nine organizer-format slides | Rendered both decks and PDFs. Organizer copies use the college and team filename. Public copies omit student numbers. | Use the correctly named organizer copy for upload. |
| Demo | 48-second public walkthrough and 59-second local backup | Both decode under three minutes. The public version uses real output and original graphics without benchmark pixels. | Use the public version for sharing. |
| Archive | Eight-output staging ZIP passed CRC and extraction | Exact model hashes and upstream locations are included. Two checkpoints were supplied separately during the reproduction check. | Confirm organizer ID, output count and distribution rights before official packaging. |
| Team and submission | Rajasthani confirmed. Google Form pending. | Team confirmation in the release brief. No issued Samsung ID was found in the supplied reference material. | Confirm whether Samsung requires a separate ID and complete the form. |
