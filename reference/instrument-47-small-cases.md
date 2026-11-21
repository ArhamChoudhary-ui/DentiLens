# Validation small-tool cases

Values in detector columns are nearest retained IoU / its score. An asterisk marks a different best-overlap proposal before cross-window merging. Visibility and occlusion are unrecorded by the source. These visual notes are single-review observations. Full model probabilities, coordinates, pre-merge evidence and the contact sheet are generated locally.

| # | Source frame | Source label | Box px | px² | Aspect | Laplacian var. | Full | ROI | Tiles | Visual review |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 101972-133548 | dental_explorer | 13 × 108 | 1404 | 8.3 | 3016 | 0.101 / 0.0010 | 0.178 / 0.0016 | 0.121 / 0.0035 | Thin shaft beside a glove and brightly lit mouth. The working tip has little separation from glare. |
| 2 | 101972-137784 | dental_mirror | 14 × 98 | 1372 | 7.0 | 4691 | 0.252 / 0.0045 | 0.517 / 0.0014 | 0.348 / 0.0031 | Long reflective shaft by gloved fingers. The visible crop gives little mirror-head evidence. |
| 3 | 101972-139050 | dental_mirror | 58 × 15 | 870 | 3.9 | 1130 | 0.138 / 0.0047 | 0.138 / 0.0047 | 0.149 / 0.0063 * | Short horizontal metallic shape against the glove. Its head and edges are hard to distinguish. |
| 4 | 101972-20682 | dental_explorer | 10 × 64 | 640 | 6.4 | 4576 | 0.061 / 0.0017 | 0.055 / 0.0040 * | 0.082 / 0.0030 | Nearly vertical thin shaft surrounded by glove and overexposed highlights. |
| 5 | 101972-20832 | dental_mirror | 14 × 66 | 924 | 4.7 | 1762 | 0.062 / 0.0120 | 0.109 / 0.0400 | 0.113 / 0.0558 | Thin reflective edge next to a broad highlight on the glove. The mirror head is unclear. |
| 6 | 101972-59334 | dental_mirror | 14 × 165 | 2310 | 11.8 | 320 | 0.121 / 0.0228 | 0.121 / 0.0228 | 0.142 / 0.0121 | Very long narrow held object at a wider camera distance. The annotated part has low contrast. |
| 7 | 101972-61242 | unknown | 13 × 60 | 780 | 4.6 | 415 | 0.113 / 0.0013 | 0.113 / 0.0013 | 0.157 / 0.0041 | Narrow bright region near bottles and gloves. The source marks instrument identity unknown. |
| 8 | 101972-64650 | dental_forceps | 12 × 68 | 816 | 5.7 | 253 | 0.416 / 0.0040 | 0.416 / 0.0040 | 0.423 / 0.0027 | Glove-side narrow object. Jaws are hard to separate from the surrounding glove in the boxed area. |
| 9 | 101994-11286 | dental_explorer | 10 × 134 | 1340 | 13.4 | 2135 | 0.049 / 0.0210 | 0.125 / 0.0010 | 0.398 / 0.0036 | Long shaft aligned with gloved fingers above the mouth. Tip details are difficult to verify. |
| 10 | 102073-40338 | unknown | 16 × 29 | 464 | 1.8 | 6390 | 0.071 / 0.0051 | 0.047 / 0.0057 * | 0.167 / 0.0021 * | Specular region in very bright oral context. The nominal 16 px case needs annotation review. |
| 11 | 102073-40896 | unknown | 14 × 44 | 616 | 3.1 | 462 | 0.032 / 0.0085 | 0.032 / 0.0085 | 0.152 / 0.0011 | Dark narrow object against a glove. Source identity remains unknown. |
| 12 | 102073-43008 | dental_mirror | 11 × 107 | 1177 | 9.7 | 566 | 0.182 / 0.0106 | 0.205 / 0.0018 | 0.222 / 0.0017 | Thin nearly vertical shaft between gloves. The small boxed crop lacks a distinct mirror head. |
