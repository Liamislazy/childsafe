# Out-of-domain test on Open Images V7 crops

Open Images *test* split crops (never used for training): 4069 crops, 2866 of the ChildSUn classes and 1203 'other' objects. ChildSUn test = canonical group split (checksum 42e3fb2547c5).
Shared classes in Open Images: few scissors (5), heaters (10) and sockets (34); no baby items.

## Summary

| | standard (ChildSUn) | robust (ChildSUn) | robust + Open Images + other |
|---|---|---|---|
| classes | 10 | 10 | 11 |
| ChildSUn acc | 92.1% | 92.1% | 90.7% |
| ChildSUn unsafe recall | 98.7% | 96.1% | 92.9% |
| Open Images acc (9 shared classes) | 74.0% | 73.1% | 87.4% |
| Open Images unsafe recall | 89.1% | 84.6% | 86.2% |
| other: called unsafe | 19.5% | 13.5% | 0.7% |
| other: predicted 'other' | - | - | 91.6% |
| pens: called unsafe | 72.8% | 60.9% | 4.3% |

'called unsafe' = calibrated P(unsafe) >= 0.5 and confidence >= 0.7 (below that it is 'uncertain', which the alert layer treats as caution).

## Share of 'other' objects called unsafe, by object

| | n | standard (ChildSUn) | robust (ChildSUn) | robust + Open Images + other |
|---|---|---|---|---|
| Book | 120 | 32.5% | 23.3% | 3.3% |
| Coffee cup | 150 | 8.7% | 1.3% | 0.0% |
| Glasses | 120 | 15.0% | 11.7% | 0.0% |
| Human hand | 300 | 19.0% | 14.3% | 0.0% |
| Mobile phone | 250 | 8.0% | 5.2% | 0.0% |
| Mug | 142 | 7.7% | 2.1% | 0.7% |
| Pen | 92 | 72.8% | 60.9% | 4.3% |
| Remote control | 18 | 22.2% | 16.7% | 0.0% |
| Toothbrush | 11 | 45.5% | 9.1% | 0.0% |
