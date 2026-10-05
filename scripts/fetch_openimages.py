"""Download object crops from Open Images V7 (validation + test splits) for:
  - the ChildSUn classes Open Images also has (out-of-domain test / extra training data)
  - an "other" set of everyday objects not in ChildSUn (pen, phone, mug, hand...)

Only the needed images are downloaded; each is cropped and only the 224x224 crop is kept.
Attribution (author, license, original URL) is saved to manifest.csv.
Annotations: CC BY 4.0. Images: Flickr, per-image licenses listed in the manifest.

    python scripts/fetch_openimages.py                  # both splits
    python scripts/fetch_openimages.py --splits test    # just the OOD test set

Output: data/external/openimages/<split>/<category>/<ImageID>_<label>.jpg
Use: Open Images "test" = out-of-domain test set; "validation" = extra training pool.
"""
import argparse
import io
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


def select_boxes(meta_dir: Path, split: str, seed: int = 0) -> pd.DataFrame:
    cls = pd.read_csv(meta_dir / "class-descriptions-boxable.csv", header=None, names=["mid", "name"])
    mid2name = dict(zip(cls.mid, cls.name))
    missing = set(MAPPING) - set(cls.name)
    if missing:
        print("[warn] classes not in Open Images:", missing)
    b = pd.read_csv(meta_dir / f"{split}-annotations-bbox.csv",
                    usecols=["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", "IsGroupOf", "IsDepiction"])
    b["oi_class"] = b.LabelName.map(mid2name)
    b = b[b.oi_class.isin(MAPPING) & (b.IsGroupOf == 0) & (b.IsDepiction == 0)]
    b = b[(b.XMax - b.XMin) * (b.YMax - b.YMin) > MIN_REL_AREA]
    imgs = pd.read_csv(meta_dir / f"{split}-images-with-rotation.csv",
                       usecols=["ImageID", "OriginalURL", "License", "Author", "Rotation"])
    imgs = imgs[imgs.Rotation.fillna(0) == 0]   # skip rotated images rather than guess the box frame
    b = b.merge(imgs, on="ImageID")
    b["area"] = (b.XMax - b.XMin) * (b.YMax - b.YMin)
    # one crop per (image, class): the largest box
    b = b.sort_values("area", ascending=False).drop_duplicates(["ImageID", "oi_class"])
    b = b.sample(frac=1, random_state=seed)
    rank = b.groupby("oi_class").cumcount()
    b = b[rank < b.oi_class.map(lambda c: CAP.get(c, DEFAULT_CAP))]
    b["category"] = b.oi_class.map(MAPPING)
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
    try:
        with urllib.request.urlopen(IMG_URL.format(split=split, image_id=image_id), timeout=30) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
    except Exception as e:
        return [(r, p, f"download failed: {type(e).__name__}") for r, p in targets]
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
    args = ap.parse_args()
    out_root = Path(args.out)
    meta_dir = out_root / "meta"
    fetch_meta(meta_dir)

    records = []
    for split in args.splits:
        sel = select_boxes(meta_dir, split)
        print(f"{split}: {len(sel)} crops from {sel.ImageID.nunique()} images")
        print(sel.groupby("category").size().to_string())
        jobs = [(iid, list(g.itertuples()), split, out_root) for iid, g in sel.groupby("ImageID")]
        with ThreadPoolExecutor(args.workers) as ex:
            for res in tqdm(ex.map(process_image, jobs), total=len(jobs), desc=f"{split} images"):
                for r, p, status in res:
                    records.append({"split": split, "category": r.category, "oi_class": r.oi_class, "image_id": r.ImageID,
                                    "path": p.relative_to(out_root).as_posix() if status in ("ok", "exists") else "",
                                    "status": status, "license": r.License, "author": r.Author, "original_url": r.OriginalURL})
    man = pd.DataFrame(records)
    man.to_csv(out_root / "manifest.csv", index=False)
    print(man.groupby(["split", "category", "status"]).size().unstack(fill_value=0).to_string())
    print(f"manifest: {out_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
