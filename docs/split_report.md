# Split report (A1)

Images: 5353. Dihedral hashing: True.

## Threshold sensitivity

| pHash thr | dHash thr | dup pairs | groups | largest group | images in groups >1 |
|---|---|---|---|---|---|
| 4 | 1 | 26 | 5327 | 2 | 52 |
| 6 | 3 | 26 | 5327 | 2 | 52 |
| 8 | 5 | 42 | 5311 | 2 | 84 |
| 10 | 7 | 133 | 5230 | 7 | 212 |
| 12 | 9 | 432 | 4981 | 57 | 497 |
| 14 | 11 | 1564 | 4476 | 691 | 1013 |

## Source photos from filenames

Distinct source photos: 1358 (paper abstract: ~1,358 raw). Sources with several augmented copies: 999.
pHash/dHash alone puts all copies of a source in one group for 0/999 sources (0.0%). Final groups = union of filename groups and hash groups.


## Chosen thresholds: pHash <= 8, dHash <= 5

Near-duplicate pairs: 42; groups: 1323; largest: 20; images sharing a group: 5046 (94.3%).

## Split sizes

| split | group split | random split |
|---|---|---|
| train | 3751 | 3747 |
| val | 813 | 803 |
| test | 789 | 803 |

## Cross-split near-duplicates (share of test images with a near-duplicate in train)

- random split: 1.0%
- group split: 0.0% (should be 0)

Share of test images whose source photo (filename) also has a copy in train:

- random split: 91.8%
- group split: 0.0%

Class balance of test split (group):

```
category
toy          240
socket       155
fork         100
knife         60
babyitems     55
ball          55
fruit         54
scissors      35
heater        30
spoon          5
```

Check `outputs/dedup/pairs_montage_*.jpg` to confirm the pairs really are duplicates / augmentations.