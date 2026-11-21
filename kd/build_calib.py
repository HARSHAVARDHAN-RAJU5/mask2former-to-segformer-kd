"""Phone-like images for NPU conversion and on-phone testing.

  make  (where the dataset is):  phone-like 512x512 PNGs (centred 4:3 cut -> 640x480 -> nearest 512x512)
        calib/  : N random TRAINING photos + all frames in --extra (e.g. Indian frames) -> calibration
        eval/   : first M VALIDATION photos, plus eval_labels/ (31-class label of the same centred 4:3 cut,
                  full size) -> accuracy check on the phone (never used for calibration)
  raw   (where QAIRT runs; needs only numpy + PIL): PNGs -> float32 NHWC .raw in [0,1] + input_list.txt

Usage:
  python3 kd/build_calib.py make --data <mapillary_vistas> --extra <frames> --out <dir> [--n-train 176] [--n-val 200]
  python3 kd/build_calib.py raw  --pngs <dir of PNGs> --out <dir>
"""

import argparse
import os
import random
import sys

import numpy as np
from PIL import Image

SIDE = 512


def make(a):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kd.dataset import CAMERA, box, cut_4_3, dataset_lut

    def phone(path):
        img = Image.open(path).convert("RGB")
        u, f = cut_4_3(*img.size, random_pos=False)
        cam = img.resize(CAMERA, Image.BILINEAR, box=box(u, f, img.size))
        return cam.resize((SIDE, SIDE), Image.NEAREST), (u, f)

    for d in ("calib", "eval", "eval_labels"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    train_dir = os.path.join(a.data, "training/images")
    picks = random.Random(0).sample(sorted(os.listdir(train_dir)), a.n_train)
    for n in picks:
        phone(os.path.join(train_dir, n))[0].save(os.path.join(a.out, "calib", "map_" + n[:-4] + ".png"))
    extra = sorted(os.listdir(a.extra)) if a.extra else []
    for n in extra:
        phone(os.path.join(a.extra, n))[0].save(os.path.join(a.out, "calib", "ind_" + os.path.splitext(n)[0] + ".png"))

    lut = dataset_lut(a.data)
    val_dir, lab_dir = os.path.join(a.data, "validation/images"), os.path.join(a.data, "validation/v1.2/labels")
    for n in sorted(os.listdir(val_dir))[:a.n_val]:
        img, (u, f) = phone(os.path.join(val_dir, n))
        img.save(os.path.join(a.out, "eval", n[:-4] + ".png"))
        lab = Image.open(os.path.join(lab_dir, n[:-4] + ".png"))
        x0, _, x1, _ = box(u, f, lab.size)
        Image.fromarray(lut[np.array(lab)][:, round(x0):round(x1)]).save(os.path.join(a.out, "eval_labels", n[:-4] + ".png"))
    print(f"calib: {len(picks)} train + {len(extra)} extra; eval: {a.n_val} val photos + labels -> {a.out}")


def raw(a):
    os.makedirs(a.out, exist_ok=True)
    names = sorted(f for f in os.listdir(a.pngs) if f.endswith(".png"))
    lines = []
    for n in names:
        x = np.asarray(Image.open(os.path.join(a.pngs, n)).convert("RGB"), dtype=np.float32)[None] / 255.0
        assert x.shape == (1, SIDE, SIDE, 3), f"{n}: {x.shape}"
        p = os.path.abspath(os.path.join(a.out, n[:-4] + ".raw"))
        x.tofile(p)
        lines.append(p)
    with open(os.path.join(a.out, "input_list.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} raw files (float32 1x{SIDE}x{SIDE}x3) + input_list.txt -> {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--data", required=True)
    m.add_argument("--extra", default="")
    m.add_argument("--out", required=True)
    m.add_argument("--n-train", type=int, default=176)
    m.add_argument("--n-val", type=int, default=200)
    r = sub.add_parser("raw")
    r.add_argument("--pngs", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    make(a) if a.cmd == "make" else raw(a)


if __name__ == "__main__":
    main()
