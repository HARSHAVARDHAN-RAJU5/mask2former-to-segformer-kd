"""The teacher (Mask2Former Swin-L, Mapillary v1.2): loading, preprocessing and 31-class probabilities.

Shared by teacher_check.py and teacher_cache.py so both run the teacher the same way.
Uses only the existing Hugging Face cache (offline): nothing is downloaded.
"""

import os

os.environ.setdefault("HF_HOME", "/mnt/nunnari3/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import Mask2FormerForUniversalSegmentation  # noqa: E402

from kd import classes as C  # noqa: E402

TEACHER = "facebook/mask2former-swin-large-mapillary-vistas-semantic"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEFAULT_SIZE = "1024"  # chosen after teacher_check: short side 1024, shape kept


def target_size(w, h, size=DEFAULT_SIZE):
    """(width, height) the teacher sees, rounded to a multiple of 32.
    "384" = squash to 384x384; "1024" = short side 1024, shape kept."""
    if size == "384":
        return 384, 384
    s = int(size) / min(w, h)
    return round(w * s / 32) * 32, round(h * s / 32) * 32


def preprocess(img, size=DEFAULT_SIZE):
    """PIL RGB image -> normalised tensor (3, H, W)."""
    w, h = target_size(*img.size, size)
    x = torch.from_numpy(np.array(img.resize((w, h), Image.BILINEAR))).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


def load(device="cuda"):
    """Teacher model and its 65 -> 31 merge matrix."""
    model = Mask2FormerForUniversalSegmentation.from_pretrained(TEACHER).to(device).eval()
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    return model, C.teacher_merge_matrix(names).to(device), names


@torch.no_grad()
def probs(model, x, matrix):
    """x: (3, H, W) preprocessed tensor on the GPU.
    Returns our 31-class probabilities (31, H/4, W/4), adding up to 1 per pixel."""
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(pixel_values=x[None])
    cls = out.class_queries_logits[0].float().softmax(-1)[:, :-1]  # (queries, 65), drop "no object"
    msk = out.masks_queries_logits[0].float().sigmoid()  # (queries, H/4, W/4)
    per_pixel = torch.einsum("qc,qhw->chw", cls, msk)  # (65, H/4, W/4)
    return C.merge_teacher_scores(per_pixel, matrix)
