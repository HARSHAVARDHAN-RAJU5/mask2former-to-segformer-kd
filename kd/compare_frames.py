"""Compare models on unlabeled frames (e.g. Indian street frames). No training.

Models:
  teacher        Mask2Former Swin-L Mapillary (31 classes), short side 1024
  kd_student     saved SegFormer-B1, alpha 0.5 (31 classes)
  base_student   saved SegFormer-B1, alpha 0   (31 classes)
  b1_cityscapes  nvidia/segformer-b1-finetuned-cityscapes-1024-1024 (19 classes), the students' start point
  b0_ade_phone   nvidia/segformer-b0-finetuned-ade-512-512 (150 classes), what the phone runs today

Every small model gets the phone input: centred 4:3 cut -> 640x480 -> nearest-neighbour 512x512.
All answers are converted to the app's 15 navigation classes (engine/core/semseg.cpp + nav_classes.h)
so they can be compared; the phone B0 uses the app's own ADE20K table.

Per frame it reports how sure the teacher is (noise check) and how much each model agrees with the
teacher. There are no human labels here, so the teacher is the reference, not the truth.

Usage (on the server):
  python3 kd/compare_frames.py --frames <folder of images> --models <saved_models folder> --out <folder>
"""

import argparse
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "0"  # B0 ADE may need a one-time download
os.environ.setdefault("HF_HOME", "/mnt/nunnari3/hf_cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import classes as C  # noqa: E402
from kd import teacher as T  # noqa: E402
from kd.dataset import CAMERA, box, cut_4_3, normalize  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from transformers import SegformerForSemanticSegmentation  # noqa: E402

SIDE = 512

# app navigation classes (engine/core/semseg.cpp)
NAV = ["obstacle", "floor", "wall", "door", "stairs", "ceiling", "person", "vehicle",
       "road", "sidewalk", "terrain", "water", "sky", "vegetation", "building"]
NAV_COLORS = np.array([[255, 140, 0], [0, 200, 80], [120, 120, 180], [255, 220, 0], [255, 0, 0],
                       [200, 200, 200], [160, 80, 255], [0, 90, 255], [90, 90, 90], [0, 200, 200],
                       [140, 170, 50], [0, 0, 160], [135, 206, 235], [30, 120, 60], [110, 60, 200]], np.uint8)
N = {n: i for i, n in enumerate(NAV)}
WALKABLE = [N["floor"], N["road"], N["sidewalk"], N["terrain"]]  # nav_classes.h navToSemRole

# our 31 classes -> nav (display / comparison only)
OURS_TO_NAV = {
    "road": "road", "sidewalk": "sidewalk", "crosswalk": "road", "curb": "obstacle", "curb cut": "sidewalk",
    "terrain": "terrain", "rail track": "road", "pothole": "obstacle", "manhole": "road",
    "building": "building", "wall": "wall", "fence": "wall", "barrier": "obstacle", "pole": "obstacle",
    "traffic light": "obstacle", "traffic sign": "obstacle", "bench": "obstacle", "trash can": "obstacle",
    "street object": "obstacle", "person": "person", "rider": "person", "animal": "obstacle",
    "car": "vehicle", "bus": "vehicle", "truck": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
    "other vehicle": "vehicle", "vegetation": "vegetation", "sky": "sky", "water": "water",
}
CITYSCAPES_TO_NAV = {
    "road": "road", "sidewalk": "sidewalk", "building": "building", "wall": "wall", "fence": "wall",
    "pole": "obstacle", "traffic light": "obstacle", "traffic sign": "obstacle", "vegetation": "vegetation",
    "terrain": "terrain", "sky": "sky", "person": "person", "rider": "person", "car": "vehicle",
    "truck": "vehicle", "bus": "vehicle", "train": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
}
# ADE20K index -> nav, copied from the app (semseg.cpp AdeMap); unlisted = obstacle
ADE_SETS = {
    "floor": [3, 28, 54], "wall": [0, 8, 18, 22, 27, 32, 42, 63, 100, 144], "door": [14, 58],
    "stairs": [53, 59, 96, 121], "ceiling": [5], "person": [12],
    "vehicle": [20, 76, 80, 83, 90, 102, 103, 116, 127], "road": [6, 91], "sidewalk": [11, 52],
    "terrain": [9, 13, 29, 46, 94], "water": [21, 26, 60, 109, 113, 128], "sky": [2],
    "vegetation": [4, 17, 66, 72], "building": [1, 25, 48, 79, 84],
}


def lut_by_name(id2label, table):
    lut = torch.zeros(len(id2label), dtype=torch.long)
    for i in range(len(id2label)):
        name = id2label[i]
        assert name in table, f"no nav mapping for class '{name}'"
        lut[i] = N[table[name]]
    return lut


def ade_lut():
    lut = torch.full((150,), N["obstacle"], dtype=torch.long)
    for nav, ids in ADE_SETS.items():
        lut[ids] = N[nav]
    return lut


def phone_input(img):
    """Centred 4:3 cut -> 640x480 (camera) -> nearest-neighbour 512x512 (the app's step)."""
    u, f = cut_4_3(*img.size, random_pos=False)
    cam = img.resize(CAMERA, Image.BILINEAR, box=box(u, f, img.size))
    return cam, normalize(cam.resize((SIDE, SIDE), Image.NEAREST))


def small_model_nav(model, x, lut):
    """Run a SegFormer on the 512x512 input; nav class map at camera size (480, 640)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        logits = model(pixel_values=x[None].cuda()).logits
    up = F.interpolate(logits.float(), size=(CAMERA[1], CAMERA[0]), mode="bilinear", align_corners=False)
    return lut.cuda()[up[0].argmax(0)]


def student(path):
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-cityscapes-1024-1024", num_labels=C.NUM_CLASSES,
        id2label=dict(enumerate(C.NAMES)), label2id={n: i for i, n in enumerate(C.NAMES)},
        ignore_mismatched_sizes=True)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    return m.cuda().eval()


def panel(nav_map, title, size=(480, 360)):
    img = Image.fromarray(NAV_COLORS[nav_map.cpu().numpy()]).resize(size, Image.NEAREST)
    return label(img, title)


def label(img, title):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 8 + 7 * len(title), 16], fill="black")
    d.text((4, 2), title, fill="white")
    return img


def legend(path):
    img = Image.new("RGB", (170, 20 * len(NAV) + 10), "white")
    d = ImageDraw.Draw(img)
    for i, n in enumerate(NAV):
        d.rectangle([5, 5 + 20 * i, 25, 19 + 20 * i], fill=tuple(int(v) for v in NAV_COLORS[i]))
        d.text((32, 6 + 20 * i), n, fill="black")
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--models", required=True, help="folder with mapillary_kd_a0.5/ and mapillary_kd_a0/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    legend(os.path.join(a.out, "legend.png"))

    teacher, matrix, _ = T.load()
    ours_lut = lut_by_name(dict(enumerate(C.NAMES)), OURS_TO_NAV)
    models = {
        "kd_student": (student(os.path.join(a.models, "mapillary_kd_a0.5", "best.pt")), ours_lut),
        "base_student": (student(os.path.join(a.models, "mapillary_kd_a0", "best.pt")), ours_lut),
    }
    b1 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
    models["b1_cityscapes"] = (b1.cuda().eval(), lut_by_name(b1.config.id2label, CITYSCAPES_TO_NAV))
    b0 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    models["b0_ade_phone"] = (b0.cuda().eval(), ade_lut())

    names = sorted(f for f in os.listdir(a.frames) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rows, walk_iou = [], {k: [] for k in models}
    for name in names:
        img = Image.open(os.path.join(a.frames, name)).convert("RGB")
        cam, x = phone_input(img)

        # teacher on the same centred 4:3 cut, at its own size (short side 1024)
        u, f = cut_4_3(*img.size, random_pos=False)
        cut = img.crop(tuple(round(v) for v in box(u, f, img.size)))
        p = T.probs(teacher, T.preprocess(cut).cuda(), matrix)  # (31, h/4, w/4)
        p = F.interpolate(p[None], size=(CAMERA[1], CAMERA[0]), mode="bilinear", align_corners=False)[0]
        conf, top = p.max(0)
        t_nav = ours_lut.cuda()[top]
        t_walk = torch.isin(t_nav, torch.tensor(WALKABLE, device="cuda"))

        row = {"frame": name, "teacher_conf": conf.mean().item(), "teacher_sure70": (conf >= 0.7).float().mean().item()}
        panels = [label(cam.resize((480, 360)), "photo (4:3 cut)"), panel(t_nav, "teacher (Mask2Former)")]
        for k, (m, lut) in models.items():
            nav = small_model_nav(m, x, lut)
            row[k] = (nav == t_nav).float().mean().item()
            walk = torch.isin(nav, torch.tensor(WALKABLE, device="cuda"))
            union = (walk | t_walk).sum().item()
            walk_iou[k].append((walk & t_walk).sum().item() / union if union else float("nan"))
            panels.append(panel(nav, k))
        rows.append(row)

        sheet = Image.new("RGB", (480 * 3, 360 * 2), "white")
        for i, pnl in enumerate(panels):
            sheet.paste(pnl, (480 * (i % 3), 360 * (i // 3)))
        sheet.save(os.path.join(a.out, os.path.splitext(name)[0] + ".jpg"), quality=88)

    keys = list(models)
    print("\nPer frame: teacher mean confidence, % pixels teacher >= 70% sure, nav agreement with teacher")
    print(f"{'frame':40s} {'t_conf':>6s} {'t>=70':>6s} " + " ".join(f"{k:>13s}" for k in keys))
    for r in sorted(rows, key=lambda r: r["teacher_conf"]):
        print(f"{r['frame'][:40]:40s} {r['teacher_conf']:6.2f} {r['teacher_sure70'] * 100:5.0f}% "
              + " ".join(f"{r[k] * 100:12.1f}%" for k in keys))
    print(f"\n{'AVERAGE':40s} {np.mean([r['teacher_conf'] for r in rows]):6.2f} "
          f"{np.mean([r['teacher_sure70'] for r in rows]) * 100:5.0f}% "
          + " ".join(f"{np.mean([r[k] for r in rows]) * 100:12.1f}%" for k in keys))
    print(f"{'walkable IoU vs teacher':40s} {'':6s} {'':6s} "
          + " ".join(f"{np.nanmean(walk_iou[k]) * 100:12.1f}%" for k in keys))


if __name__ == "__main__":
    main()
