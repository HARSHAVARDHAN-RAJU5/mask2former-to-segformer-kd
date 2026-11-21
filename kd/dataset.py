"""Training and validation data for the student.

Train example = photo + human label (31 classes, 255 = ignore) + teacher map (31-class probabilities).
The teacher map is rebuilt from the cached top 5 (other classes 0, rescaled to add up to 1), then
photo, label and map get the SAME 4:3 cut, stretch, zoom, cut and flip. The map is always exactly
1/4 of the photo's size and crops start at multiples of 4, so map pixel (i, j) covers photo pixels
(4i..4i+3, 4j..4j+3).

Phone match: the camera frame is 640x480 (4:3) and the app stretches it to 512x512 with
nearest-neighbour (engine/core/semseg.cpp in the app). So:
  - photos wider than 4:3 are first cut to 4:3 (train: random horizontal position; val: centred);
    narrower photos are left as they are,
  - then the frame is stretched to a square (train: side 512 x random zoom; val: 512x512),
  - validation does camera-size 640x480 first, then the app's nearest-neighbour stretch to 512x512.
"""

import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ColorJitter

from kd import classes as C

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CAMERA = (640, 480)  # phone camera frame (4:3)
ASPECT = CAMERA[0] / CAMERA[1]


def normalize(img):
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


def dataset_lut(data):
    names = [l["readable"] for l in json.load(open(os.path.join(data, "config_v1.2.json")))["labels"]]
    return C.gt_lookup(names)


def load_teacher_map(path):
    """Cached top 5 -> full (31, h, w) probability map: other classes 0, rescaled to add up to 1."""
    z = np.load(path)
    idx = torch.from_numpy(z["idx"].astype(np.int64))
    prob = torch.from_numpy(z["prob"].astype(np.float32))
    full = torch.zeros((C.NUM_CLASSES,) + tuple(idx.shape[1:]))
    full.scatter_(0, idx, prob)
    return full / full.sum(0, keepdim=True).clamp_min(1e-6)


def cut_4_3(w, h, random_pos):
    """Horizontal 4:3 cut as fractions of the width: (start u, width f).
    Photos not wider than 4:3 (2% tolerance) are kept whole: (0, 1)."""
    if w / h <= ASPECT * 1.02:
        return 0.0, 1.0
    f = ASPECT / (w / h)
    return (random.uniform(0.0, 1.0 - f) if random_pos else (1.0 - f) / 2), f


def box(u, f, size):
    """Pixel box (x0, y0, x1, y1) for the cut [u, u+f] of the width, full height, in an image of `size`."""
    w, h = size
    return (u * w, 0.0, (u + f) * w, float(h))


def cut_and_resize_map(tmap, u, f, out_h, out_w):
    """Take the same width fraction [u, u+f] of the teacher map and resize to (out_h, out_w), bilinear.
    Uses exact (fractional) positions, like PIL's resize(box=...) does for the photo."""
    jj = (torch.arange(out_w) + 0.5) / out_w  # output pixel centres, 0..1
    ii = (torch.arange(out_h) + 0.5) / out_h
    gx = (u + f * jj) * 2 - 1  # grid_sample coordinates, -1..1
    gy = ii * 2 - 1
    grid = torch.stack(torch.meshgrid(gy, gx, indexing="ij")[::-1], -1)[None]  # (1, out_h, out_w, 2) as (x, y)
    return F.grid_sample(tmap[None], grid, mode="bilinear", padding_mode="border", align_corners=False)[0]


class TrainSet(torch.utils.data.Dataset):
    """Returns (photo (3,crop,crop), label (crop,crop), teacher map (31,crop/4,crop/4) float16)."""

    def __init__(self, data, cache, side=512, crop=512, scale=(0.5, 1.5), flip=True, jitter=True):
        assert crop % 4 == 0 and side % 4 == 0, "side and crop must be multiples of 4"
        self.img_dir = os.path.join(data, "training/images")
        self.lab_dir = os.path.join(data, "training/v1.2/labels")
        self.cache = cache
        self.names = sorted(f[:-4] for f in os.listdir(self.img_dir))
        missing = [n for n in self.names if not os.path.exists(os.path.join(cache, n + ".npz"))]
        assert not missing, f"{len(missing)} photos have no teacher cache file, e.g. {missing[:3]}"
        self.lut = dataset_lut(data)
        self.side, self.crop, self.scale, self.flip = side, crop, scale, flip
        self.jitter = ColorJitter(0.2, 0.2, 0.2) if jitter else None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(os.path.join(self.img_dir, name + ".jpg"))
        u, f = cut_4_3(*img.size, random_pos=True)  # same cut for photo, label and map

        # stretch to a square like the app, with random zoom; the photo is always exactly 4x the map
        s = random.uniform(*self.scale)
        mh = mw = round(self.side / 4 * s)
        ph, pw = mh * 4, mw * 4

        img.draft("RGB", (math.ceil(pw / f), ph))  # faster JPEG decoding when shrinking a lot
        img = img.convert("RGB").resize((pw, ph), Image.BILINEAR, box=box(u, f, img.size))
        if self.jitter:
            img = self.jitter(img)  # colour change: photo only
        lab = Image.fromarray(self.lut[np.array(Image.open(os.path.join(self.lab_dir, name + ".png")))])
        lab = lab.resize((pw, ph), Image.NEAREST, box=box(u, f, lab.size))
        tmap = load_teacher_map(os.path.join(self.cache, name + ".npz"))  # whole photo, (31, hm, wm)

        x = normalize(img)
        y = torch.from_numpy(np.array(lab)).long()
        t = cut_and_resize_map(tmap, u, f, mh, mw)

        # pad up to the crop size if the zoomed photo is smaller (label 255, so CE and KD skip it)
        c, cm = self.crop, self.crop // 4
        if ph < c or pw < c:
            pad_h, pad_w = max(0, c - ph), max(0, c - pw)  # multiples of 4
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
            y = F.pad(y, (0, pad_w, 0, pad_h), value=C.IGNORE_ID)
            t = F.pad(t, (0, pad_w // 4, 0, pad_h // 4), value=0.0)
            mh, mw = mh + pad_h // 4, mw + pad_w // 4

        # crop: same place for all three; chosen in map units, so the photo crop starts at a multiple of 4
        oy, ox = random.randint(0, mh - cm), random.randint(0, mw - cm)
        x = x[:, oy * 4:oy * 4 + c, ox * 4:ox * 4 + c]
        y = y[oy * 4:oy * 4 + c, ox * 4:ox * 4 + c]
        t = t[:, oy:oy + cm, ox:ox + cm]

        if self.flip and random.random() < 0.5:
            x, y, t = x.flip(-1), y.flip(-1), t.flip(-1)
        return x.contiguous(), y.contiguous(), t.half().contiguous()


class ValSet(torch.utils.data.Dataset):
    """Returns (photo as the phone sees it, label at full size of the centred 4:3 cut).
    Photo: centred 4:3 cut -> camera size 640x480 -> nearest-neighbour stretch to side x side (the app's step).
    The student's square answer is stretched back and scored at the label's full size."""

    def __init__(self, data, side=512, limit=0):
        self.img_dir = os.path.join(data, "validation/images")
        self.lab_dir = os.path.join(data, "validation/v1.2/labels")
        names = sorted(f[:-4] for f in os.listdir(self.img_dir))
        self.names = names[:limit] if limit else names
        self.lut = dataset_lut(data)
        self.side = side

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(os.path.join(self.img_dir, name + ".jpg"))
        u, f = cut_4_3(*img.size, random_pos=False)
        img.draft("RGB", (math.ceil(CAMERA[0] / f), CAMERA[1]))
        cam = img.convert("RGB").resize(CAMERA, Image.BILINEAR, box=box(u, f, img.size))
        x = normalize(cam.resize((self.side, self.side), Image.NEAREST))
        lab = Image.open(os.path.join(self.lab_dir, name + ".png"))
        x0, _, x1, _ = box(u, f, lab.size)
        y = self.lut[np.array(lab)][:, round(x0):round(x1)]
        return x, torch.from_numpy(y.astype(np.int64))
