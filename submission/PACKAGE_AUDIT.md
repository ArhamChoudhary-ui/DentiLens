# Samsung Track 4 package audit

Team Rajasthani · Vellore Institute of Technology, Vellore

| Component | Status | Evidence |
| --- | --- | --- |
| Branch and source | Found | Existing inference lives in `track4/code`. The packaging work started at `4468a73` on `xr-4-video-intelligence`. |
| Entry point | Verified | The packaged `run.sh` accepts two arguments, rejects missing arguments, and reproduces a result from a working directory and input path containing spaces. |
| Inference and dependencies | Verified | `code/` and the pinned Python 3.11 requirements reproduced an existing output with verified checkpoints. |
| Raw checkpoints | Found locally | The ZIP includes class metadata, SHA-256 hashes and upstream acquisition instructions. Checkpoint binaries remain local pending redistribution clearance. |
| Outputs | Verified locally, incomplete | The archive snapshot contains 30 of 187 locally verified predictions, including the original eight. The acquisition batch is separate. |
| Official validator | Missing | The supplied kit, repository and Git history have no organizer `validate.py`. The archive includes the separate local checker. |
| Organizer TEAM_ID | Missing | Rajasthani is the confirmed team name. Confirm the separate issued ID before naming the final archive or regenerating its JSON files. |
| Disclosure | Found | [AI disclosure](../AI_DISCLOSURE.md). |
| Presentation and demo | Verified | The organizer-named PPTX and PDF and the public 48-second demo are kept beside the ZIP. |
| Extra top-level files | Not applicable | The ZIP contains the six organizer entries. Presentation, demo, disclosure and reports are separate. |

The staged ZIP uses the root folder `Rajasthani_Staging`, with
`run.sh`, `code/`, `weights/`, `requirements.txt`, `outputs/`
and `README.md` in that order.

Rebuild a snapshot from a directory of locally valid outputs:

```bash
python track4/scripts/stage_submission.py /path/to/outputs /path/to/new-staging
```

Use the [submission checklist](FINAL_SUBMISSION_CHECKLIST.md) to verify the
remaining organizer requirements.
