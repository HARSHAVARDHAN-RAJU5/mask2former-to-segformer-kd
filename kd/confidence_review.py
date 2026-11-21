"""Test B: what do the confidence filters do on Indian frames? (no human labels: judged by eye)

Reads the teacher cache (top-5 classes + probabilities) for unlabeled frames and reports:
  - how frames spread over average-confidence bands
  - how much data survives each frame threshold x pixel threshold
  - pictures: for a few frames per band: photo | teacher's top class | pixels kept at 0.8 | kept at 0.9
    (removed pixels greyed out)

Usage (on the server):
  python3 kd/confidence_review.py --images <frames> --cache <teacher cache> --out <folder> [--per-band 8]
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import classes as C  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

FRAME_BANDS = [0.0, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
FRAME_THRESH = [0.0, 0.7, 0.75, 0.8, 0.85]
PIXEL_THRESH = [0.0, 0.7, 0.8, 0.9]
PANEL_W = 384


def palette():
    rng = np.random.RandomState(3)  # same colours as teacher_check.py / train.py line-up
    return rng.randint(40, 255, size=(256, 3)).astype(np.uint8)


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(text), 16], fill="black")
    d.text((4, 2), text, fill="white")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-band", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    names = sorted(f[:-4] for f in os.listdir(a.cache) if f.endswith(".npz"))
    stats = {}
    for n in names:
        conf = np.load(os.path.join(a.cache, n + ".npz"))["prob"][0].astype(np.float32)  # top-1 probability
        stats[n] = (conf.mean(), [(conf >= t).mean() for t in PIXEL_THRESH])
    means = np.array([s[0] for s in stats.values()])
    print(f"{len(names)} frames in the cache\n")
    print("Frames by average teacher confidence:")
    for i in range(len(FRAME_BANDS) - 1):
        m = (means >= FRAME_BANDS[i]) & (means < FRAME_BANDS[i + 1])
        print(f"  {FRAME_BANDS[i]:.2f}-{min(FRAME_BANDS[i + 1], 1):.2f}: {m.sum():6d} frames ({m.mean() * 100:5.1f}%)")

    print("\nShare of all Indian pixels kept (frame threshold x pixel threshold):")
    print(f"{'frame >=':>9s} {'frames kept':>12s} " + " ".join(f"{'px>=' + str(t):>9s}" for t in PIXEL_THRESH))
    for ft in FRAME_THRESH:
        keep = [s for s in stats.values() if s[0] >= ft]
        cols = [np.sum([s[1][j] for s in keep]) / len(stats) * 100 for j in range(len(PIXEL_THRESH))]
        print(f"{ft:9.2f} {len(keep) / len(stats) * 100:11.1f}% " + " ".join(f"{c:8.1f}%" for c in cols))

    pal = palette()
    rng = random.Random(0)
    for i in range(len(FRAME_BANDS) - 1):
        lo, hi = FRAME_BANDS[i], FRAME_BANDS[i + 1]
        band = [n for n in names if lo <= stats[n][0] < hi]
        for k, n in enumerate(rng.sample(band, min(a.per_band, len(band)))):
            z = np.load(os.path.join(a.cache, n + ".npz"))
            top, conf = z["idx"][0], z["prob"][0].astype(np.float32)
            h, w = top.shape
            ph = round(PANEL_W * h / w)
            img = Image.open(os.path.join(a.images, n + ".jpg")).convert("RGB")
            photo = img.resize((PANEL_W, ph))
            grey = np.asarray(photo.convert("L").convert("RGB"), dtype=np.uint8) // 3
            panels = [label(photo.copy(), f"photo  avg conf {stats[n][0]:.2f}"),
                      label(Image.fromarray(pal[top]).resize((PANEL_W, ph), Image.NEAREST), "teacher top class")]
            for t in (0.8, 0.9):
                keep = np.asarray(Image.fromarray((conf >= t).astype(np.uint8) * 255).resize((PANEL_W, ph), Image.NEAREST)) > 0
                vis = np.where(keep[..., None], np.asarray(photo), grey)
                panels.append(label(Image.fromarray(vis.astype(np.uint8)), f"kept px>={t} ({(conf >= t).mean() * 100:.0f}%)"))
            sheet = Image.new("RGB", (PANEL_W * 4, ph), "white")
            for j, p in enumerate(panels):
                sheet.paste(p, (PANEL_W * j, 0))
            sheet.save(os.path.join(a.out, f"band_{lo:.2f}-{min(hi, 1):.2f}_{k}.jpg"), quality=85)
    print(f"\nPictures -> {a.out} (up to {a.per_band} per band; grey = pixels removed)")


if __name__ == "__main__":
    main()
