"""Download object crops from Open Images V7 (validation + test splits) for:
  - the ChildSUn classes Open Images also has (out-of-domain test / extra training data)
  - an "other" set of everyday objects not in ChildSUn (pen, phone, mug, hand...)

Only the needed images are downloaded; each is cropped and only the 224x224 crop is kept.
Attribution (author, license, original URL) is saved to manifest.csv.
Annotations: CC BY 4.0. Images: Flickr, per-image licenses listed in the manifest.

    python scripts/fetch_openimages.py                  # validation + test (small, ~40 classes)
    python scripts/fetch_openimages.py --splits test    # just the OOD test set
    python scripts/fetch_openimages.py --splits train --dry-run   # ~100 classes from train: counts + size only
    python scripts/fetch_openimages.py --splits train             # then download them

Output: data/external/openimages/<split>/<category>/<ImageID>_<label>.jpg
Use: Open Images "test" = out-of-domain test set; "validation" = extra training pool.
"""
import argparse
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from childsafe.config import ROOT

BASE = "https://storage.googleapis.com/openimages"
META = {"classes": "v5/class-descriptions-boxable.csv",
        "validation_boxes": "v5/validation-annotations-bbox.csv", "test_boxes": "v5/test-annotations-bbox.csv",
        "validation_images": "2018_04/validation/validation-images-with-rotation.csv",
        "test_images": "2018_04/test/test-images-with-rotation.csv"}
TRAIN_META = ["v6/oidv6-train-annotations-bbox.csv", "2018_04/train/train-images-boxable-with-rotation.csv"]
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"

# Open Images class -> our category. ChildSUn "fruits" also contains vegetables, so a few are included.
# No Open Images class matches ChildSUn "baby items". "Power plugs and sockets" also contains plugs.
MAPPING = {
    "Knife": "knife", "Kitchen knife": "knife", "Fork": "fork", "Spoon": "spoon", "Scissors": "scissors",
    "Power plugs and sockets": "socket", "Heater": "heater",
    "Ball": "ball", "Football": "ball", "Volleyball (Ball)": "ball", "Rugby ball": "ball", "Tennis ball": "ball",
    "Golf ball": "ball", "Cricket ball": "ball",
    "Toy": "toy", "Teddy bear": "toy", "Doll": "toy",
    "Fruit": "fruit", "Apple": "fruit", "Banana": "fruit", "Orange": "fruit", "Strawberry": "fruit", "Grape": "fruit",
    "Pear": "fruit", "Lemon": "fruit", "Mango": "fruit", "Peach": "fruit", "Pineapple": "fruit",
    "Watermelon": "fruit", "Cucumber": "fruit", "Bell pepper": "fruit", "Carrot": "fruit",
    # "other": objects outside the 10 ChildSUn classes. NOT a claim that they are safe for children
    # (a pen cap is a choking hazard); it means "not one of the trained classes".
    "Pen": "other", "Mobile phone": "other", "Mug": "other", "Coffee cup": "other", "Human hand": "other",
    "Toothbrush": "other", "Remote control": "other", "Book": "other", "Glasses": "other",
}
# Extra "other" classes used for the *train* split only (the OOD test keeps the 9 'other' classes above, so test
# numbers stay comparable). Everyday household objects; deliberately EXCLUDED because they are real hazards and
# must not be learned as harmless: balloons, plastic bags, glasses/vases (breakable), watches (button cells),
# coins, rings, earrings, candy, light bulbs, kettles, stoves, ovens, irons, tools and weapons.
TRAIN_OTHER = [
    "Pencil case", "Sunglasses", "Bottle", "Bowl", "Plate", "Saucer", "Drinking straw", "Towel", "Pillow", "Chair",
    "Table", "Couch", "Bed", "Lamp", "Television", "Computer keyboard", "Computer mouse", "Laptop", "Tablet computer",
    "Headphones", "Camera", "Clock", "Alarm clock", "Handbag", "Backpack", "Umbrella", "Hat", "Sun hat", "Sock",
    "Footwear", "Boot", "Sandal", "Scarf", "Glove", "Toilet paper", "Paper towel", "Soap dispenser", "Picture frame",
    "Flowerpot", "Houseplant", "Box", "Envelope", "Calculator", "Light switch", "Door handle", "Drawer", "Shelf",
    "Bookcase", "Curtain", "Bread", "Cookie", "Muffin", "Cake", "Sandwich", "Musical keyboard", "Guitar", "Drum",
    "Cat", "Dog", "Measuring cup", "Ladle",
]
TRAIN_MAPPING = {**MAPPING, **{c: "other" for c in TRAIN_OTHER}}
TRAIN_CAP_TARGET = 1500    # per Open Images class mapped to a ChildSUn class
TRAIN_CAP_FRUIT = 250      # per fruit/vegetable class (there are 15)
TRAIN_CAP_OTHER = 150      # per "other" class (~70 classes)
TRAIN_CAP_SPECIAL = {"Ball": 500, "Football": 500, "Toy": 800, "Doll": 800, "Teddy bear": 800}  # keep ball/toy from dominating
FRUITS = {c for c, v in MAPPING.items() if v == "fruit"}
MB_PER_IMAGE = 0.25        # rough average download size, for the dry-run estimate

# cap crops per Open Images class and split, so hands/fruit do not swamp everything
CAP = {"Human hand": 300, "Fruit": 200, "Toy": 300, "Glasses": 120, "Coffee cup": 150, "Book": 120, "Mobile phone": 250}
DEFAULT_CAP = 400
MIN_REL_AREA = 0.004   # skip tiny boxes
MIN_SIDE_PX = 48       # skip crops smaller than this in the original image
MARGIN = 1.2           # square crop = 1.2 x the larger box side
OUT_SIZE = 224


def fetch_meta(meta_dir: Path):
    meta_dir.mkdir(parents=True, exist_ok=True)
    for rel in META.values():
        dest = meta_dir / Path(rel).name
        if not dest.exists():
            print("downloading", rel)
            urllib.request.urlretrieve(f"{BASE}/{rel}", dest)


def mapping_for(split: str) -> dict:
    return TRAIN_MAPPING if split == "train" else MAPPING


def cap_for(split: str, oi_class: str) -> int:
    if split != "train":
        return CAP.get(oi_class, DEFAULT_CAP)
    if oi_class in TRAIN_CAP_SPECIAL:
        return TRAIN_CAP_SPECIAL[oi_class]
    if TRAIN_MAPPING[oi_class] == "other":
        return TRAIN_CAP_OTHER
    return TRAIN_CAP_FRUIT if oi_class in FRUITS else TRAIN_CAP_TARGET


def select_boxes(meta_dir: Path, split: str, seed: int = 0) -> pd.DataFrame:
    mapping = mapping_for(split)
    cls = pd.read_csv(meta_dir / "class-descriptions-boxable.csv", header=None, names=["mid", "name"])
    mid2name = dict(zip(cls.mid, cls.name))
    missing = set(mapping) - set(cls.name)
    if missing:
        print("[warn] classes not in Open Images:", missing)
    boxes_file = "oidv6-train-annotations-bbox.csv" if split == "train" else f"{split}-annotations-bbox.csv"
    images_file = "train-images-boxable-with-rotation.csv" if split == "train" else f"{split}-images-with-rotation.csv"
    wanted = {m for m, n in mid2name.items() if n in mapping}
    parts = []
    for chunk in pd.read_csv(meta_dir / boxes_file, chunksize=2_000_000,   # the train file is 2.3 GB
                             usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"]):
        chunk = chunk[chunk.LabelName.isin(wanted) & (chunk.IsGroupOf == 0) & (chunk.IsDepiction == 0)]
        parts.append(chunk[(chunk.XMax - chunk.XMin) * (chunk.YMax - chunk.YMin) > MIN_REL_AREA])
    b = pd.concat(parts, ignore_index=True)
    b["oi_class"] = b.LabelName.map(mid2name)
    imgs = pd.read_csv(meta_dir / images_file,
                       usecols=["ImageID", "OriginalURL", "License", "Author", "Rotation"])
    imgs = imgs[imgs.Rotation.fillna(0) == 0]   # skip rotated images rather than guess the box frame
    b = b.merge(imgs, on="ImageID")
    b["area"] = (b.XMax - b.XMin) * (b.YMax - b.YMin)
    # one crop per (image, class): the largest box
    b = b.sort_values("area", ascending=False).drop_duplicates(["ImageID", "oi_class"])
    b = b.sample(frac=1, random_state=seed)
    rank = b.groupby("oi_class").cumcount()
    b = b[rank < b.oi_class.map(lambda c: cap_for(split, c))]
    b["category"] = b.oi_class.map(mapping)
    b["split"] = split
    return b


def crop_one(img: Image.Image, r) -> Image.Image | None:
    w, h = img.size
    x0, x1, y0, y1 = r.XMin * w, r.XMax * w, r.YMin * h, r.YMax * h
    side = max(x1 - x0, y1 - y0)
    if side < MIN_SIDE_PX:
        return None
    side *= MARGIN
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))
    return img.crop(box).resize((OUT_SIZE, OUT_SIZE), Image.Resampling.BICUBIC)


def process_image(args):
    image_id, rows, split, out_root = args
    targets = [(r, out_root / split / r.category / f"{image_id}_{r.oi_class.replace(' ', '_')}.jpg") for r in rows]
    if all(p.exists() for _, p in targets):
        return [(r, p, "exists") for r, p in targets]
    img, err = None, None
    for attempt in range(4):   # network drops happen on long runs: retry with backoff
        try:
            with urllib.request.urlopen(IMG_URL.format(split=split, image_id=image_id), timeout=30) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
            break
        except Exception as e:
            err = e
            time.sleep(2 * 2 ** attempt)
    if img is None:
        return [(r, p, f"download failed: {type(err).__name__}") for r, p in targets]
    out = []
    for r, p in targets:
        c = crop_one(img, r)
        if c is None:
            out.append((r, p, "too small"))
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        c.save(p, quality=92)
        out.append((r, p, "ok"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["test", "validation"])
    ap.add_argument("--out", default=str(ROOT / "data" / "external" / "openimages"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true", help="only count crops and estimate the download size")
    args = ap.parse_args()
    out_root = Path(args.out)
    meta_dir = out_root / "meta"
    fetch_meta(meta_dir)
    if "train" in args.splits:
        for rel in TRAIN_META:
            dest = meta_dir / Path(rel).name
            if not dest.exists():
                print("downloading", rel, "(train metadata, ~2.9 GB in total)")
                urllib.request.urlretrieve(f"{BASE}/{rel}", dest)

    records = []
    for split in args.splits:
        sel = select_boxes(meta_dir, split)
        print(f"{split}: {len(sel)} crops from {sel.ImageID.nunique()} images")
        print(sel.groupby("category").size().to_string())
        n_img = sel.ImageID.nunique()
        print(f"estimated download: ~{n_img * MB_PER_IMAGE / 1024:.1f} GB ({n_img} images); "
              f"crops kept on disk: ~{len(sel) * 0.02:.0f} MB")
        if args.dry_run:
            print(sel.groupby(["category", "oi_class"]).size().to_string())
            continue
        jobs = [(iid, list(g.itertuples()), split, out_root) for iid, g in sel.groupby("ImageID")]
        with ThreadPoolExecutor(args.workers) as ex:
            for res in tqdm(ex.map(process_image, jobs), total=len(jobs), desc=f"{split} images"):
                for r, p, status in res:
                    records.append({"split": split, "category": r.category, "oi_class": r.oi_class, "image_id": r.ImageID,
                                    "path": p.relative_to(out_root).as_posix() if status in ("ok", "exists") else "",
                                    "status": status, "license": r.License, "author": r.Author, "original_url": r.OriginalURL})
    if args.dry_run:
        return
    man = pd.DataFrame(records)
    old = out_root / "manifest.csv"
    if old.exists():   # keep rows of splits not fetched in this run
        prev = pd.read_csv(old)
        man = pd.concat([prev[~prev.split.isin(args.splits)], man], ignore_index=True)
    man.to_csv(out_root / "manifest.csv", index=False)
    print(man.groupby(["split", "category", "status"]).size().unstack(fill_value=0).to_string())
    print(f"manifest: {out_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
