# Complete-video instrument benchmark

The benchmark streams original source frames at the current detector
cadence of round(FPS/5), rounded to at least one native frame. Its
reported durations weight each sampled observation until the next sample
or end of file. They are sampled instrument-seconds, not exact sub-frame
visibility. The last visible sample alone never establishes an exit.

Detection AP uses score-ranked normalized boxes and IoU 0.50–0.95.
The runtime detector has one class. Per-instrument scores are localization
recall on that class's reference boxes. Manual-crop classification currently
scores unambiguous mirror and explorer crops independently of the detector.
Tracking first pairs boxes at each sample and then assigns trajectories
globally for IDF1. Unmatched detections, duplicate physical tracks, missed
visibility, and correctly named visibility have separate counts. No HOTA
score is emitted.

The CLI keeps raw observations, identities and tracks for each resolution
and for released and experimental classification. Geometry and shadow
output never write back to application examinations. A live review label
is never used as model output or evaluation truth.
