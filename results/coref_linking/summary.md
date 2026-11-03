# Identity linking on TAB test

Pooled over annotators (counts summed, rate = numerator/denominator). Detected PERSON spans are matched to the same annotator's DIRECT or QUASI PERSON mentions by overlap; spans matching none are excluded.

## gliner+presidio_structured

| Measure | Resolver off | Resolver on |
| --- | --- | --- |
| Entities with 2+ detected mentions and 2+ surface forms: all mentions under one placeholder | 0.034 (6/179) | 0.799 (143/179) |
| Entities with 2+ detected mentions: all mentions under one placeholder | 0.497 (181/364) | 0.874 (318/364) |
| Placeholders holding mentions of 2+ different people | 0.003 (8/3053) | 0.062 (163/2619) |
| Detected PERSON spans matched / unmatched | 3781 / 8228 (summed over annotators) | 3781 / 8228 (summed over annotators) |

## spacy_lg+presidio_structured

| Measure | Resolver off | Resolver on |
| --- | --- | --- |
| Entities with 2+ detected mentions and 2+ surface forms: all mentions under one placeholder | 0.063 (17/269) | 0.747 (201/269) |
| Entities with 2+ detected mentions: all mentions under one placeholder | 0.370 (161/435) | 0.832 (362/435) |
| Placeholders holding mentions of 2+ different people | 0.009 (27/3157) | 0.073 (189/2604) |
| Detected PERSON spans matched / unmatched | 3744 / 986 (summed over annotators) | 3744 / 986 (summed over annotators) |
