"""The 31 student classes and how the Mapillary v1.2 classes map onto them.

One place for the table, used by the teacher check, the teacher cache and training.
Everything is matched by class NAME (never by assumed ID) and checked when the tables are built.

Rules:
- Human labels: Mapillary ID -> our ID through a lookup table; void classes -> 255 (ignore).
- Teacher scores: the probabilities of merged classes are ADDED into the target class.
  Car Mount and Ego Vehicle are left out, and the remaining 31 are rescaled to add up to 1.
"""

import numpy as np
import torch

IGNORE_ID = 255

# (our class name, Mapillary v1.2 source classes); list order = our ID
CLASSES = [
    ("road", ["Road", "Service Lane", "Parking", "Bike Lane", "Lane Marking - General"]),
    ("sidewalk", ["Sidewalk", "Pedestrian Area"]),
    ("crosswalk", ["Crosswalk - Plain", "Lane Marking - Crosswalk"]),
    ("curb", ["Curb"]),
    ("curb cut", ["Curb Cut"]),
    ("terrain", ["Terrain", "Sand", "Snow", "Mountain"]),
    ("rail track", ["Rail Track"]),
    ("pothole", ["Pothole"]),
    ("manhole", ["Manhole", "Catch Basin"]),
    ("building", ["Building", "Bridge", "Tunnel"]),
    ("wall", ["Wall"]),
    ("fence", ["Fence", "Guard Rail"]),
    ("barrier", ["Barrier"]),
    ("pole", ["Pole", "Utility Pole", "Street Light", "Traffic Sign Frame"]),
    ("traffic light", ["Traffic Light"]),
    ("traffic sign", ["Traffic Sign (Front)", "Traffic Sign (Back)"]),
    ("bench", ["Bench"]),
    ("trash can", ["Trash Can"]),
    ("street object", ["Bike Rack", "Fire Hydrant", "Junction Box", "Mailbox", "Phone Booth",
                       "CCTV Camera", "Billboard", "Banner"]),
    ("person", ["Person"]),
    ("rider", ["Bicyclist", "Motorcyclist", "Other Rider"]),
    ("animal", ["Ground Animal", "Bird"]),
    ("car", ["Car"]),
    ("bus", ["Bus"]),
    ("truck", ["Truck"]),
    ("motorcycle", ["Motorcycle"]),
    ("bicycle", ["Bicycle"]),
    ("other vehicle", ["Other Vehicle", "Wheeled Slow", "Caravan", "Trailer", "On Rails", "Boat"]),
    ("vegetation", ["Vegetation"]),
    ("sky", ["Sky"]),
    ("water", ["Water"]),
]
IGNORE_SOURCES = ["Car Mount", "Ego Vehicle", "Unlabeled"]

NAMES = [name for name, _ in CLASSES]
NUM_CLASSES = len(CLASSES)  # 31


def _source_to_id():
    """Mapillary class name -> our ID (or IGNORE_ID). Fails if a name is used twice."""
    out = {}
    for our_id, (_, sources) in enumerate(CLASSES):
        for s in sources:
            assert s not in out, f"'{s}' is used twice in the class table"
            out[s] = our_id
    for s in IGNORE_SOURCES:
        assert s not in out, f"'{s}' is used twice in the class table"
        out[s] = IGNORE_ID
    return out


def _check_covered(names, what):
    """Every name in `names` must be in the table, and every table name must exist somewhere."""
    table = _source_to_id()
    missing = [n for n in names if n not in table]
    assert not missing, f"{what} classes not in the class table: {missing}"


def gt_lookup(dataset_names):
    """Lookup table for human labels: lut[mapillary_id] = our ID (255 = ignore).

    dataset_names: class names in dataset ID order (from config_v1.2.json, 66 incl. Unlabeled).
    Use as: our_mask = lut[mapillary_mask]
    """
    _check_covered(dataset_names, "Dataset")
    table = _source_to_id()
    unknown = [s for s in table if s not in dataset_names]
    assert not unknown, f"Class table names not in the dataset: {unknown}"
    lut = np.full(256, IGNORE_ID, dtype=np.uint8)
    for mid, name in enumerate(dataset_names):
        lut[mid] = table[name]
    return lut


def teacher_merge_matrix(teacher_names):
    """0/1 matrix (65 x 31): row = teacher class, column = our class it is added into.

    teacher_names: the teacher's id2label in ID order (65 classes, no Unlabeled).
    Rows for Car Mount / Ego Vehicle are all zero (left out).
    """
    _check_covered(teacher_names, "Teacher")
    table = _source_to_id()
    m = torch.zeros(len(teacher_names), NUM_CLASSES)
    for tid, name in enumerate(teacher_names):
        if table[name] != IGNORE_ID:
            m[tid, table[name]] = 1.0
    # every one of our classes must get at least one teacher class
    empty = [NAMES[i] for i in range(NUM_CLASSES) if m[:, i].sum() == 0]
    assert not empty, f"Our classes with no teacher class: {empty}"
    return m


def merge_teacher_scores(per_pixel, matrix, eps=1e-6):
    """Teacher per-pixel scores (65, H, W) -> our class probabilities (31, H, W), adding up to 1."""
    merged = torch.einsum("chw,ck->khw", per_pixel, matrix.to(per_pixel))
    return merged / merged.sum(0, keepdim=True).clamp_min(eps)
