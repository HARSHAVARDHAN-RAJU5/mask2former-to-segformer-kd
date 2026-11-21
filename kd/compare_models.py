"""Compare students on the held-back Indian videos (no human labels: the teacher is the reference).

For every test frame (phone-like input: centred 4:3 cut -> 640x480 -> nearest 512x512) each model's 128x128
answer is compared with the teacher's cached answer for the same centred 4:3 cut, ONLY on cells where the
teacher is >= --conf sure (the same rule as training). Reports per model:
  - agreement with the teacher on those cells
  - per-class IoU against the teacher (31 classes)
Also prints the Mapillary val per-class IoU from each run's best_iou.json when --runs are given.

Usage (on the server):
  python3 kd/compare_models.py --models old=<best.pt> new=<best.pt> --images <frames> --cache <teacher cache> \
      --list <file names> [--conf 0.9] [--runs old=<run folder> new=<run folder>]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import classes as C  # noqa: E402
from kd.dataset import CAMERA, box, cut_4_3, cut_and_resize_map, load_teacher_conf, load_teacher_map, normalize  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import SegformerForSemanticSegmentation  # noqa: E402

os.environ.setdefault("HF_HOME", "/mnt/nunnari3/hf_cache")
SIDE, OUT, NC = 512, 128, C.NUM_CLASSES


def student(path):
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-cityscapes-1024-1024", num_labels=NC,
        id2label=dict(enumerate(C.NAMES)), label2id={n: i for i, n in enumerate(C.NAMES)},
        ignore_mismatched_sizes=True)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return m.cuda().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path/to/best.pt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--conf", type=float, default=0.9)
    ap.add_argument("--runs", nargs="*", default=[], help="name=run folder (for Mapillary val per class)")
    a = ap.parse_args()

    models = {k: student(v) for k, v in (s.split("=", 1) for s in a.models)}
    names = [l.strip()[:-4] for l in open(a.list) if l.strip()]
    names = [n for n in names if os.path.exists(os.path.join(a.cache, n + ".npz"))]
    conf_mat = {k: np.zeros((NC, NC), np.int64) for k in models}
    cells_used = cells_all = 0
    for n in names:
        img = Image.open(os.path.join(a.images, n + ".jpg")).convert("RGB")
        u, f = cut_4_3(*img.size, random_pos=False)
        cam = img.resize(CAMERA, Image.BILINEAR, box=box(u, f, img.size))
        x = normalize(cam.resize((SIDE, SIDE), Image.NEAREST))[None].cuda()
        path = os.path.join(a.cache, n + ".npz")
        maps = cut_and_resize_map(torch.cat([load_teacher_map(path), load_teacher_conf(path)]), u, f, OUT, OUT)
        t_top, t_ok = maps[:NC].argmax(0).numpy(), (maps[NC] >= a.conf).numpy()
        cells_used += t_ok.sum()
        cells_all += t_ok.size
        for k, m in models.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                pred = m(pixel_values=x).logits[0].argmax(0).cpu().numpy()
            conf_mat[k] += np.bincount(t_top[t_ok] * NC + pred[t_ok], minlength=NC * NC).reshape(NC, NC)

    print(f"{len(names)} held-back Indian frames; cells where the teacher is >= {a.conf} sure: "
          f"{cells_used / cells_all * 100:.1f}%\n")
    ious = {}
    for k, cm in conf_mat.items():
        inter = np.diag(cm)
        union = cm.sum(0) + cm.sum(1) - inter
        ious[k] = np.where(cm.sum(1) > 0, inter / np.maximum(union, 1), np.nan)
        print(f"{k:8s} agreement with teacher {inter.sum() / cm.sum() * 100:5.1f}%   "
              f"mean IoU vs teacher {np.nanmean(ious[k]):.4f}")

    runs = dict(s.split("=", 1) for s in a.runs)
    val = {k: json.load(open(os.path.join(p, "best_iou.json"))) for k, p in runs.items()}
    keys = list(models)
    head = " ".join(f"{'IND ' + k:>10s}" for k in keys) + "".join(f" {'MAP ' + k:>10s}" for k in val)
    print(f"\n{'class':15s} {head}   (IND = vs teacher on Indian test frames, MAP = Mapillary val vs human labels)")
    for i, cname in enumerate(C.NAMES):
        row = " ".join(f"{ious[k][i]:10.3f}" if not np.isnan(ious[k][i]) else f"{'-':>10s}" for k in keys)
        row += "".join(f" {v['per_class_iou'][cname]:10.3f}" if v["per_class_iou"][cname] is not None else f" {'-':>10s}"
                       for v in val.values())
        print(f"{cname:15s} {row}")
    for k, v in val.items():
        print(f"Mapillary val {k}: mIoU {v['miou']:.4f} (step {v['step']})")


if __name__ == "__main__":
    main()
