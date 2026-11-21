"""Test A: when can the teacher's confidence be trusted? (uses Mapillary val, which has human labels)

The teacher runs on N validation photos exactly as for the cache (whole photo, short side 1024).
Its top class and confidence per pixel are compared with the human label (31 classes, ignore skipped):
  - per pixel:  accuracy for each confidence band, and the share of pixels in that band
  - per photo:  average confidence vs the photo's pixel accuracy

Usage (on the server):
  python3 kd/teacher_calibration.py --data <mapillary_vistas> [--n 300]
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import classes as C  # noqa: E402
from kd import teacher as T  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

PIXEL_BANDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
PHOTO_BANDS = [0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()

    names = [l["readable"] for l in json.load(open(os.path.join(a.data, "config_v1.2.json")))["labels"]]
    lut = C.gt_lookup(names)
    model, matrix, _ = T.load()
    files = sorted(os.listdir(os.path.join(a.data, "validation/images")))
    picks = random.Random(1).sample(files, a.n)

    right = np.zeros(len(PIXEL_BANDS) - 1)
    total = np.zeros(len(PIXEL_BANDS) - 1)
    photos = []
    for n in picks:
        img = Image.open(os.path.join(a.data, "validation/images", n)).convert("RGB")
        p = T.probs(model, T.preprocess(img).cuda(), matrix)  # (31, h, w)
        conf, top = p.max(0)
        conf, top = conf.cpu().numpy(), top.cpu().numpy()
        lab = Image.open(os.path.join(a.data, "validation/v1.2/labels", n[:-4] + ".png"))
        lab = lut[np.array(lab.resize((top.shape[1], top.shape[0]), Image.NEAREST))]
        v = lab != C.IGNORE_ID
        ok = (top == lab) & v
        band = np.digitize(conf, PIXEL_BANDS) - 1
        right += np.bincount(band[ok], minlength=len(total))[:len(total)]
        total += np.bincount(band[v], minlength=len(total))[:len(total)]
        photos.append((conf.mean(), ok.sum() / max(v.sum(), 1)))

    print(f"Teacher on {a.n} Mapillary val photos (short side 1024, 31 classes)\n")
    print("Per pixel:  confidence band -> how often the top class equals the human label")
    print(f"{'band':>12s} {'share of pixels':>16s} {'accuracy':>9s}")
    for i in range(len(total)):
        lo, hi = PIXEL_BANDS[i], min(PIXEL_BANDS[i + 1], 1.0)
        if total[i]:
            print(f"{lo:5.2f}-{hi:4.2f} {total[i] / total.sum() * 100:15.1f}% {right[i] / total[i] * 100:8.1f}%")
    print("\nIf we keep only pixels at or above a threshold:")
    print(f"{'threshold':>9s} {'pixels kept':>12s} {'accuracy of kept':>17s}")
    for i, t in enumerate(PIXEL_BANDS[:-1]):
        kept_r, kept_t = right[i:].sum(), total[i:].sum()
        print(f"{t:9.2f} {kept_t / total.sum() * 100:11.1f}% {kept_r / kept_t * 100:16.1f}%")

    pc = np.array(photos)
    print("\nPer photo:  average confidence band -> photos in band, their average pixel accuracy")
    for i in range(len(PHOTO_BANDS) - 1):
        lo, hi = PHOTO_BANDS[i], min(PHOTO_BANDS[i + 1], 1.0)
        m = (pc[:, 0] >= lo) & (pc[:, 0] < PHOTO_BANDS[i + 1])
        if m.any():
            print(f"{lo:5.2f}-{hi:4.2f} {m.sum():5d} photos  accuracy {pc[m, 1].mean() * 100:5.1f}%  (worst {pc[m, 1].min() * 100:5.1f}%)")
    print(f"\ncorrelation(photo confidence, photo accuracy) = {np.corrcoef(pc[:, 0], pc[:, 1])[0, 1]:.2f}")


if __name__ == "__main__":
    main()
