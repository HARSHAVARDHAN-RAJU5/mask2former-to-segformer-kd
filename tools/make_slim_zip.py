"""Make a smaller Mapillary Vistas zip with only what v1.2 training needs.

Keeps: train/val images, train/val v1.2 semantic labels, config_v1.2.json, LICENSE, README.
Drops: testing images (no labels), v1.2 instances/panoptic, everything v2.0.
Files inside keep the same paths. Stored without compression (JPEG/PNG don't shrink further).
The original zip is only read, never changed.

Usage:
  python tools/make_slim_zip.py <source.zip> <output.zip>
"""

import os
import shutil
import sys
import zipfile

from tqdm import tqdm

KEEP_PREFIXES = (
    "training/images/",
    "validation/images/",
    "training/v1.2/labels/",
    "validation/v1.2/labels/",
)
KEEP_FILES = {"config_v1.2.json", "LICENSE", "README"}


def wanted(name):
    return name in KEEP_FILES or name.startswith(KEEP_PREFIXES)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1:]
    if os.path.exists(out):
        sys.exit(f"{out} already exists. Delete it first if you want to rebuild it.")
    tmp = out + ".partial"  # renamed at the end, so a half-built zip is never mistaken for a finished one

    with zipfile.ZipFile(src) as zin:
        items = [i for i in zin.infolist() if not i.is_dir() and wanted(i.filename)]
        total = sum(i.file_size for i in items)
        print(f"Keeping {len(items)} files, {total / 1e9:.2f} GB")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zout, \
                tqdm(total=total, unit="B", unit_scale=True, desc="building") as bar:
            for info in items:
                new = zipfile.ZipInfo(info.filename, info.date_time)
                new.compress_type = zipfile.ZIP_STORED
                with zin.open(info) as fi, zout.open(new, "w", force_zip64=True) as fo:
                    shutil.copyfileobj(fi, fo, 4 * 1024 * 1024)
                bar.update(info.file_size)

    # Quick check: same file list and every file's CRC matches the original
    with zipfile.ZipFile(tmp) as z:
        assert sorted(z.namelist()) == sorted(i.filename for i in items), "file list mismatch"
        bad = z.testzip()
        assert bad is None, f"damaged file in output: {bad}"
    os.replace(tmp, out)
    print(f"Done: {out} ({os.path.getsize(out) / 1e9:.2f} GB), all {len(items)} files checked OK")


if __name__ == "__main__":
    main()
