"""Grad-CAM (Selvaraju et al., ICCV 2017) overlays for qualitative inspection.

    python -m childsafe.classifier.gradcam --run outputs/runs/efficientnet_b0_group --n 40
    python -m childsafe.classifier.gradcam --run ... --images data/ood/knife   # any folder
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from childsafe.classifier.calibration import softmax
from childsafe.classifier.models import gradcam_target_layer, load_checkpoint
from childsafe.config import load_config
from childsafe.data.catalog import IMG_EXT, source_roots
from childsafe.data.dataset import eval_transform, image_paths


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # tensor hook instead of a module backward hook: timm uses in-place activations
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor, target: int | None = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        target = int(logits.argmax(1)) if target is None else target
        logits[0, target].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach().cpu().numpy()[0], target


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--images", default=None, help="folder of images; default: random test-split images")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run = Path(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(run / "best.pt", device)
    cam = GradCAM(model, gradcam_target_layer(model, ckpt["arch"]))
    size = ckpt["img_size"]
    tf = eval_transform(size, preprocessing=ckpt.get("preprocessing", "center_crop"))
    show = T.Compose(tf.transforms[:-2])  # same spatial preprocessing as model input

    if args.images:
        paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in IMG_EXT)[: args.n]
        truth = [""] * len(paths)
    else:
        df = pd.read_csv(cfg["paths"]["splits"])
        df = df[df[f"split_{ckpt.get('split', 'group')}"] == "test"].sample(frac=1, random_state=0).head(args.n)
        paths = image_paths(df, source_roots(cfg))
        truth = list(df["category"])

    out = run / "gradcam"
    out.mkdir(exist_ok=True)
    for k, (p, t) in enumerate(zip(paths, truth)):
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device).requires_grad_(True)
        heat, logits, target = cam(x)
        prob = softmax(logits[None], ckpt["temperature"])[0]
        disp = np.asarray(show(img))
        vis = overlay(disp, heat)
        label = f"pred {ckpt['classes'][target]} {prob[target]:.2f}" + (f" | true {t}" if t else "")
        vis = cv2.putText(vis.copy(), label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        Image.fromarray(vis).save(out / f"{k:03d}_{Path(p).stem}.jpg")
    print(f"wrote {len(paths)} overlays to {out}")


if __name__ == "__main__":
    main()
