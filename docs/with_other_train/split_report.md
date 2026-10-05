# Split report (A1)

Images: 27883. Dihedral hashing: True.

## Threshold sensitivity

| pHash thr | dHash thr | dup pairs | groups | largest group | images in groups >1 |
|---|---|---|---|---|---|
| 4 | 1 | 212 | 27672 | 3 | 421 |
| 6 | 3 | 298 | 27587 | 3 | 589 |
| 8 | 5 | 406 | 27498 | 25 | 742 |
| 10 | 7 | 697 | 27343 | 65 | 971 |
| 12 | 9 | 1603 | 26896 | 155 | 1508 |
| 14 | 11 | 5164 | 25555 | 969 | 3013 |

## Source photos from filenames

Distinct source photos: 22918 (paper abstract: ~1,358 raw). Sources with several augmented copies: 1914.
pHash/dHash alone puts all copies of a source in one group for 277/1914 sources (14.5%). Final groups = union of filename groups and hash groups.


ChildSUn rows keep the split of `data/splits/splits.csv`. Extra images that are near-duplicates of a ChildSUn image follow it: 0; dropped as bridges between ChildSUn splits: 0.


## Chosen thresholds: pHash <= 8, dHash <= 5

Near-duplicate pairs: 406; groups: 22839; largest: 25; images sharing a group: 6990 (25.1%).

## Split sizes

| split | group split | random split |
|---|---|---|
| train | 19523 | 19517 |
| val | 4194 | 4183 |
| test | 4166 | 4183 |

## Cross-split near-duplicates (share of test images with a near-duplicate in train)

- random split: 1.7%
- group split: 0.0% (should be 0)

Share of test images whose source photo (filename) also has a copy in train:

- random split: 22.4%
- group split: 0.0%

Class balance of test split (group):

```
category
other        1587
toy           641
fruit         604
ball          387
fork          253
knife         197
socket        178
spoon         158
scissors       74
babyitems      55
heater         32
```

Check `outputs/dedup/pairs_montage_*.jpg` to confirm the pairs really are duplicates / augmentations.