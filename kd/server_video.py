"""Run the Mask2Former teacher on a video on the server GPU and write the same overlay video as phone_video.py.

Each frame: teacher at its normal input (short side 1024, shape kept) -> 31-class probabilities (same maths as
the teacher cache) -> class map stretched to the frame size. Same colours, legend and layout as phone_video.py,
so the two videos can be compared side by side.

Usage (on the server):
  python3 kd/server_video.py --video <in.mp4> --out <out.mp4> [--size 1024]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import teacher as T  # noqa: E402  (sets the offline cache before transformers loads)
from kd.phone_video import NC, PAL, legend  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="1024", help='"1024" = short side 1024 (normal); "384" = squashed')
    a = ap.parse_args()

    model, matrix, _ = T.load()
    cap = cv2.VideoCapture(a.video)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    leg = legend(2 * w)
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h + leg.shape[0]))
    counts, times, t0 = np.zeros(NC, np.int64), [], time.time()
    label = f"Mask2Former Swin-L on server GPU (short side {a.size})"
    for i in range(total):
        ok, f = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        torch.cuda.synchronize()
        s = time.time()
        p = T.probs(model, T.preprocess(img, a.size).cuda(), matrix)
        torch.cuda.synchronize()
        times.append(time.time() - s)
        cls = cv2.resize(p.argmax(0).cpu().numpy().astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        counts += np.bincount(cls.ravel(), minlength=NC)
        over = cv2.addWeighted(f, 0.45, PAL[cls][..., ::-1], 0.55, 0)
        cv2.putText(over, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(np.vstack([np.hstack([f, over]), leg]))
        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{total} frames, {np.mean(times[-300:]) * 1000:.0f} ms/frame, {time.time() - t0:.0f} s", flush=True)
    out.release()
    from kd import classes as C
    share = counts / counts.sum() * 100
    top = ", ".join(f"{C.NAMES[k]} {share[k]:.1f}%" for k in np.argsort(-share)[:8])
    print(f"\nGPU average {np.mean(times[1:]) * 1000:.0f} ms/frame over {len(times)} frames; total {time.time() - t0:.0f} s")
    print(f"Most common classes (share of all pixels): {top}")
    print(f"Saved {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
