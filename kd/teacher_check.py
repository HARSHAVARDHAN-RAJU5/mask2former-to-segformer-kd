"""Run the teacher (Mask2Former Swin-L, Mapillary) on a few training images and check it.

For each image and each input size:
  1. Resize the photo and run the teacher on the GPU (fp16 autocast, no gradients).
  2. Build per-pixel scores: class probabilities x mask probabilities, summed over the queries.
  3. Merge 65 teacher classes into our 31 (kd/classes.py).
  4. Compare with the human label (ignore pixels skipped) and save photo | human label | teacher.

Prints time per image, peak GPU memory, pixel accuracy and mIoU for each input size.

Usage (on the server):
  python3 kd/teacher_check.py --data <mapillary_vistas folder> --out <output folder> [--n 5]
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import classes as C  # noqa: E402
from kd import teacher as T  # noqa: E402  (sets the offline cache before transformers loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

SIZES = ["384", "1024"]


def palette():
    rng = np.random.RandomState(3)
    p = rng.randint(40, 255, size=(256, 3)).astype(np.uint8)
    p[C.IGNORE_ID] = 0  # ignore = black
    return p


def save_legend(path, pal):
    img = Image.new("RGB", (220, 20 * C.NUM_CLASSES + 20), "white")
    d = ImageDraw.Draw(img)
    for i, name in enumerate(C.NAMES + ["ignore"]):
        cid = i if i < C.NUM_CLASSES else C.IGNORE_ID
        d.rectangle([5, 3 + 20 * i, 25, 17 + 20 * i], fill=tuple(int(v) for v in pal[cid]))
        d.text((32, 4 + 20 * i), f"{cid:3d} {name}", fill="black")
    img.save(path)


def miou(conf):
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    present = union > 0
    return (inter[present] / union[present]).mean(), present.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds_names = [l["readable"] for l in json.load(open(os.path.join(a.data, "config_v1.2.json")))["labels"]]
    lut = C.gt_lookup(ds_names)
    model, matrix, t_names = T.load()
    print(f"Class tables OK: {len(ds_names)} dataset / {len(t_names)} teacher -> {C.NUM_CLASSES} ours")

    files = sorted(os.listdir(os.path.join(a.data, "training/images")))
    picks = random.Random(a.seed).sample(files, a.n)
    pal = palette()
    save_legend(os.path.join(a.out, "legend.png"), pal)

    def run(img, size):
        return T.probs(model, T.preprocess(img, size).cuda(), matrix)

    for size in SIZES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        run(Image.open(os.path.join(a.data, "training/images", picks[0])).convert("RGB"), size)  # warm-up
        conf = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
        times = []
        for name in picks:
            img = Image.open(os.path.join(a.data, "training/images", name)).convert("RGB")
            gt = lut[np.array(Image.open(os.path.join(a.data, "training/v1.2/labels", name[:-4] + ".png")))]

            torch.cuda.synchronize()
            t0 = time.time()
            probs = run(img, size)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

            # compare at the human label's full size
            up = F.interpolate(probs[None].half(), size=gt.shape, mode="bilinear", align_corners=False)
            pred = up[0].argmax(0).cpu().numpy().astype(np.uint8)
            valid = gt != C.IGNORE_ID
            conf += np.bincount(gt[valid].astype(np.int64) * C.NUM_CLASSES + pred[valid],
                                minlength=C.NUM_CLASSES ** 2).reshape(C.NUM_CLASSES, C.NUM_CLASSES)
            acc = (pred[valid] == gt[valid]).mean()
            print(f"  [{size}] {name}  {img.size[0]}x{img.size[1]} -> {T.target_size(*img.size, size)}  "
                  f"{times[-1] * 1000:.0f} ms  pixel acc {acc:.3f}")

            w = 640
            h = round(img.size[1] * w / img.size[0])
            panels = [img.resize((w, h)),
                      Image.fromarray(pal[gt]).resize((w, h), Image.NEAREST),
                      Image.fromarray(pal[pred]).resize((w, h), Image.NEAREST)]
            sheet = Image.new("RGB", (w * 3, h))
            for i, p in enumerate(panels):
                sheet.paste(p, (w * i, 0))
            sheet.save(os.path.join(a.out, f"{name[:-4]}_{size}.jpg"), quality=90)

        m, n = miou(conf)
        print(f"[{size}] avg {np.mean(times) * 1000:.0f} ms/image | peak GPU "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB | pixel acc "
              f"{np.diag(conf).sum() / conf.sum():.3f} | mIoU {m:.3f} over {n} classes present\n")


if __name__ == "__main__":
    main()
