"""Training and validation data for the student.

Train example = photo + human label (31 classes, 255 = ignore) + teacher map (31-class probabilities).
The teacher map is rebuilt from the cached top 5 (other classes 0, rescaled to add up to 1), then
photo, label and map get the SAME zoom, cut and flip. The map is always exactly 1/4 of the photo's
size and cuts start at multiples of 4, so map pixel (i, j) covers photo pixels (4i..4i+3, 4j..4j+3).

Phone match: the app stretches the whole camera frame to 512x512 (engine/core/semseg.cpp), so here
every frame is also stretched to a square (side 512 x random zoom) before the cut, and validation
stretches to 512x512.
"""

import json
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


class TrainSet(torch.utils.data.Dataset):
    """Returns (photo (3,crop,crop), label (crop,crop), teacher map (31,crop/4,crop/4) float16)."""

    def __init__(self, data, cache, side=512, crop=512, scale=(0.5, 1.5), flip=True, jitter=True):
        assert crop % 4 == 0 and side % 4 == 0, "side and crop must be multiples of 4"
        self.side = side
        self.img_dir = os.path.join(data, "training/images")
        self.lab_dir = os.path.join(data, "training/v1.2/labels")
        self.cache = cache
        self.names = sorted(f[:-4] for f in os.listdir(self.img_dir))
        missing = [n for n in self.names if not os.path.exists(os.path.join(cache, n + ".npz"))]
        assert not missing, f"{len(missing)} photos have no teacher cache file, e.g. {missing[:3]}"
        self.lut = dataset_lut(data)
        self.crop, self.scale, self.flip = crop, scale, flip
        self.jitter = ColorJitter(0.2, 0.2, 0.2) if jitter else None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        tmap = load_teacher_map(os.path.join(self.cache, name + ".npz"))  # (31, hm, wm)

        # stretch to a square like the app, with random zoom; pick the map size first,
        # the photo is always exactly 4x the map
        s = random.uniform(*self.scale)
        mh = mw = round(self.side / 4 * s)
        ph, pw = mh * 4, mw * 4

        img = Image.open(os.path.join(self.img_dir, name + ".jpg"))
        img.draft("RGB", (pw, ph))  # faster JPEG decoding when shrinking a lot
        img = img.convert("RGB").resize((pw, ph), Image.BILINEAR)
        if self.jitter:
            img = self.jitter(img)  # colour change: photo only
        lab = Image.fromarray(self.lut[np.array(Image.open(os.path.join(self.lab_dir, name + ".png")))])
        lab = lab.resize((pw, ph), Image.NEAREST)

        x = normalize(img)
        y = torch.from_numpy(np.array(lab)).long()
        t = F.interpolate(tmap[None], size=(mh, mw), mode="bilinear", align_corners=False)[0]

        # pad up to the crop size if the zoomed photo is smaller (label 255, so CE and KD skip it)
        c, cm = self.crop, self.crop // 4
        if ph < c or pw < c:
            pad_h, pad_w = max(0, c - ph), max(0, c - pw)  # multiples of 4
            x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
            y = F.pad(y, (0, pad_w, 0, pad_h), value=C.IGNORE_ID)
            t = F.pad(t, (0, pad_w // 4, 0, pad_h // 4), value=0.0)
            mh, mw = mh + pad_h // 4, mw + pad_w // 4

        # cut: same place for all three; chosen in map units, so the photo cut starts at a multiple of 4
        oy, ox = random.randint(0, mh - cm), random.randint(0, mw - cm)
        x = x[:, oy * 4:oy * 4 + c, ox * 4:ox * 4 + c]
        y = y[oy * 4:oy * 4 + c, ox * 4:ox * 4 + c]
        t = t[:, oy:oy + cm, ox:ox + cm]

        if self.flip and random.random() < 0.5:
            x, y, t = x.flip(-1), y.flip(-1), t.flip(-1)
        return x.contiguous(), y.contiguous(), t.half().contiguous()


class ValSet(torch.utils.data.Dataset):
    """Returns (photo stretched to side x side like the app, label at full size).
    The student's answer is stretched back and scored at the label's full size."""

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
        img.draft("RGB", (self.side, self.side))
        x = normalize(img.convert("RGB").resize((self.side, self.side), Image.BILINEAR))
        y = torch.from_numpy(self.lut[np.array(Image.open(os.path.join(self.lab_dir, name + ".png")))].astype(np.int64))
        return x, y
