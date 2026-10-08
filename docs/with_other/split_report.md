# Split report (A1)

Images: 7326. Dihedral hashing: True.

## Threshold sensitivity

| pHash thr | dHash thr | dup pairs | groups | largest group | images in groups >1 |
|---|---|---|---|---|---|
| 4 | 1 | 67 | 7259 | 2 | 134 |
| 6 | 3 | 87 | 7239 | 2 | 174 |
| 8 | 5 | 108 | 7218 | 2 | 216 |
| 10 | 7 | 204 | 7132 | 7 | 354 |
| 12 | 9 | 512 | 6874 | 61 | 650 |
| 14 | 11 | 1690 | 6332 | 704 | 1216 |

## Source photos from filenames

Distinct source photos: 3144 (paper abstract: ~1,358 raw). Sources with several augmented copies: 1172.
pHash/dHash alone puts all copies of a source in one group for 58/1172 sources (4.9%). Final groups = union of filename groups and hash groups.


ChildSUn rows keep the split of `data/splits/splits.csv`. Extra images that are near-duplicates of a ChildSUn image follow it: 0; dropped as bridges between ChildSUn splits: 0.


## Chosen thresholds: pHash <= 8, dHash <= 5

Near-duplicate pairs: 108; groups: 3109; largest: 20; images sharing a group: 5406 (73.8%).

## Split sizes

| split | group split | random split |
|---|---|---|
| train | 5134 | 5128 |
| val | 1109 | 1099 |
| test | 1083 | 1099 |

## Cross-split near-duplicates (share of test images with a near-duplicate in train)

- random split: 2.4%
- group split: 0.0% (should be 0)

Share of test images whose source photo (filename) also has a copy in train:

- random split: 70.2%
- group split: 0.0%

Class balance of test split (group):

```
category
toy          281
socket       157
fruit        132
other        119
fork         105
ball          84
knife         72
babyitems     55
scissors      35
heater        31
spoon         12
```

Check `outputs/dedup/pairs_montage_*.jpg` to confirm the pairs really are duplicates / augmentations.