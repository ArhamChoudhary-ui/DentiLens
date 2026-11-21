# Instrument identity audit

The current video detector proposes instrument regions. The crop models score the proposed regions, and the tracker can extend a supported identity across nearby observations. An unknown prediction stays unresolved until the model has sufficient evidence or a reviewer identifies the tool.

## Saved portrait examination

The audit reran identity and tracking with the existing 320 candidate boxes from clip 6502638, without reviewer overrides. It found 7 duplicates and accepted names on 159 of the 313 canonical observations. Of the 154 unresolved observations, the top class was unknown on 150, three fell below the class threshold, and one had no enabled class. Seven unresolved crops had a short side of 12–23 pixels after encoding on the 96-pixel classifier canvas. The remaining 147 measured at least 24 pixels on their short side.

Testing 4%, 10%, and 20% padding on the same boxes yielded 159 named observations at each setting. The app retains 10% padding and its existing confidence thresholds.

## New video checks

The same pipeline sampled both clips across their full duration at approximately 5 Hz.

| Clip | Candidate boxes | Tracks | Accepted instrument names |
| --- | ---: | ---: | ---: |
| 4489972 | 7 | 4 | 0 |
| 14195516 | 11 | 4 | 0 |

These clips provide examples to investigate. They lack complete tool annotations and verified source grouping, so these counts do not establish precision, recall, or generalization. No weights were retrained or promoted in this update.

## Review workflow

Open **Instruments**, enable model candidates, and choose **Unidentified tracks**. Select a result and open **Inspect model crop** to view the input region, its source dimensions, and the effective resolution on the classifier canvas. Use **Leave unknown** when the tool cannot be identified. That decision preserves the original model prediction and excludes the reviewed observation from confirmed exports.

The next training pass needs labeled video examples of the tools the model misses across different angles and distances, along with negative examples showing gloves, teeth, braces, and reflections. Keep full source recordings separate between training and evaluation.
