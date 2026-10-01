# ChildSUn dataset report

Root: `D:\AI_project\Project\Piise_children_safety\data\raw\childsun`

## Folder tree (image counts)

```
.: 5353
```

Non-image files by extension: {'.json': 1}

## Catalog

Labelled images: **5353**

| category | binary | count |
|---|---|---|
| babyitems | safe | 375 |
| ball | safe | 385 |
| fork | unsafe | 670 |
| fruit | safe | 359 |
| heater | unsafe | 210 |
| knife | unsafe | 415 |
| scissors | unsafe | 244 |
| socket | unsafe | 1050 |
| spoon | safe | 55 |
| toy | safe | 1590 |

Binary totals: {'safe': 2764, 'unsafe': 2589}

Image sizes (sample of 500): {(224, 224): 500}

File names matching augmentation-like patterns: 4994 (e.g. ['baby_items_10_0_1720.jpg', 'baby_items_10_0_316.jpg', 'baby_items_10_0_6049.jpg', 'baby_items_10_0_8479.jpg', 'baby_items_10_0_9975.jpg'])


## Still to check by hand

- License (see dataset-metadata.json from the download script)
- Whether augmented copies are included (compare with the ~1,358 raw / ~5,350 total in the paper abstract; see near-duplicate groups from scripts/make_splits.py)
- Safe/unsafe grouping of each category vs the paper
