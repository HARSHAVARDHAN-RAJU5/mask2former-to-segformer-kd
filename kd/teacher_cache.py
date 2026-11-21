"""Run the teacher once on every training photo and save its answers (the "teacher cache").

For each photo:
  1. Resize so the short side is 1024 and run the teacher (fp16, no gradients).
  2. Get our 31-class probabilities per pixel at 1/4 of the teacher's input size.
  3. Keep the top 5 classes per pixel and save:  idx (5,H,W) uint8,  prob (5,H,W) float16,
     orig_hw = the photo's original (height, width).
One .npz file per photo, written under a temporary name and renamed when complete.
Photos that already have a file are skipped, so a stopped run can simply be started again.
A photo that fails (broken file, GPU out of memory, ...) is skipped and written to the failed log
(name + error) instead of stopping the whole job. A restart retries it, because it has no file.

At the end it prints: time per photo, file size, estimates for the full set, how much
probability the top 5 hold, and a re-load check of a few saved files against the live teacher.

Usage (on the server):
  python3 kd/teacher_cache.py --data <mapillary_vistas folder> --out <cache folder> [--limit 100]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import teacher as T  # noqa: E402  (sets the offline cache before transformers loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

TOP_K = 5


class Photos(torch.utils.data.Dataset):
    """Opens and preprocesses photos in worker processes while the GPU is busy."""

    def __init__(self, folder, names):
        self.folder, self.names = folder, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        """Returns (tensor, name, (height, width)), or (None, name, error text) if the photo can't be read."""
        try:
            img = Image.open(os.path.join(self.folder, self.names[i])).convert("RGB")
            return T.preprocess(img), self.names[i], (img.size[1], img.size[0])
        except Exception as e:
            return None, self.names[i], f"{type(e).__name__}: {e}"


def topk(probs):
    """(31, H, W) probabilities -> top-5 class ids (uint8) and probabilities (float16), on the CPU."""
    p, i = probs.topk(TOP_K, dim=0)
    return i.to(torch.uint8).cpu().numpy(), p.half().cpu().numpy()


def save(path, idx, prob, orig_hw):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, idx=idx, prob=prob, orig_hw=np.array(orig_hw, dtype=np.int32))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="only the first N photos (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify", type=int, default=3, help="saved files to re-check against the live teacher")
    ap.add_argument("--failed", default="", help="failed-photo log (default: <out>_failed.txt next to --out)")
    ap.add_argument("--images", default="", help="photo folder (default: <data>/training/images)")
    ap.add_argument("--list", default="", help="text file of photo file names to use (default: all in the folder)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    failed_log = a.failed or a.out.rstrip("/") + "_failed.txt"

    folder = a.images or os.path.join(a.data, "training/images")
    all_names = sorted(os.listdir(folder))
    if a.list:
        wanted = {l.strip() for l in open(a.list) if l.strip()}
        all_names = [n for n in all_names if n in wanted]
    names = all_names[:a.limit] if a.limit else all_names
    todo = [n for n in names if not os.path.exists(os.path.join(a.out, n[:-4] + ".npz"))]
    print(f"{len(names)} photos selected, {len(names) - len(todo)} already cached, {len(todo)} to do")
    if not todo:
        return

    model, matrix, _ = T.load()
    loader = torch.utils.data.DataLoader(Photos(folder, todo), batch_size=None, num_workers=a.workers)

    def log_failure(name, err):
        with open(failed_log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\t{err}\n")
        tqdm.write(f"FAILED {name}: {err}")
        bar.set_postfix(failed=failed)

    t0 = time.time()
    nbytes, coverage, done, failed = 0, [], [], 0
    bar = tqdm(loader, total=len(names), initial=len(names) - len(todo), desc="caching",
               unit="photo", mininterval=2, dynamic_ncols=True)
    for x, name, info in bar:
        if x is None:  # photo couldn't be read; info = error text
            failed += 1
            log_failure(name, info)
            continue
        try:
            probs = T.probs(model, x.cuda(non_blocking=True), matrix)
            idx, prob = topk(probs)
            path = os.path.join(a.out, name[:-4] + ".npz")
            save(path, idx, prob, info)
        except Exception as e:  # includes GPU out of memory
            failed += 1
            log_failure(name, f"{type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue
        done.append(name)
        nbytes += os.path.getsize(path)
        coverage.append(float(prob.astype(np.float32).sum(0).mean()))
        bar.set_postfix(failed=failed)
    bar.close()
    elapsed = time.time() - t0

    print(f"\nFinished: {len(done)} saved, {failed} failed"
          + (f" (see {failed_log})" if failed else "") + f", {elapsed / 60:.1f} min")
    if not done:
        return
    n = len(done)
    per = elapsed / n
    print(f"  time per photo : {per:.2f} s   -> all {len(all_names)} photos: {per * len(all_names) / 3600:.1f} h")
    print(f"  file size      : {nbytes / n / 1e6:.2f} MB -> all {len(all_names)} photos: "
          f"{nbytes / n * len(all_names) / 1e9:.1f} GB")
    print(f"  top-{TOP_K} hold on average {np.mean(coverage) * 100:.2f}% of the probability "
          f"(lowest photo {np.min(coverage) * 100:.2f}%)")

    # re-load check: saved file vs live teacher on the same photo
    for name in done[:a.verify]:
        d = np.load(os.path.join(a.out, name[:-4] + ".npz"))
        img = Image.open(os.path.join(folder, name)).convert("RGB")
        live_idx, live_prob = topk(T.probs(model, T.preprocess(img).cuda(), matrix))
        same_top1 = (d["idx"][0] == live_idx[0]).mean()
        diff = np.abs(d["prob"].astype(np.float32) - live_prob.astype(np.float32)).max()
        print(f"  verify {name}: shape {d['idx'].shape}, orig {tuple(d['orig_hw'])}, "
              f"top-1 same {same_top1 * 100:.2f}%, max prob diff {diff:.4f}")


if __name__ == "__main__":
    main()
