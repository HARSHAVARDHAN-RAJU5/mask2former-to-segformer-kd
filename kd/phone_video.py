"""Run a student on the phone NPU over a whole video and write an overlay video (run on the PC, phone on adb).

Each frame goes through the app's step: frame -> (640x480 camera size) -> nearest-neighbour 512x512, RGB 0..1.
Frames are sent to the phone in chunks, run with qnn-net-run (burst, NPU time profiled), and the 128x128 class
map is stretched back to the frame size. Output video: original | overlay, with a legend of the 31 classes.

Usage:
  python kd/phone_video.py --video <in.mp4> --model b1r3_w8a16 --out <out.mp4> --work <PC temp dir>
The model must already be on the phone as /data/local/tmp/sfb1/models/<model>.bin with the QAIRT runtime there.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import classes as C  # noqa: E402

REMOTE = "/data/local/tmp/sfb1"
ENV = f"cd {REMOTE} && export LD_LIBRARY_PATH={REMOTE} ADSP_LIBRARY_PATH={REMOTE}"
SIDE, OUT, NC, CAMERA = 512, 128, C.NUM_CLASSES, (640, 480)

# readable colours (RGB) for our 31 classes
COLORS = {
    "road": (90, 90, 90), "sidewalk": (244, 35, 232), "crosswalk": (255, 255, 255), "curb": (255, 220, 0),
    "curb cut": (255, 150, 0), "terrain": (150, 170, 60), "rail track": (110, 70, 40), "pothole": (255, 0, 0),
    "manhole": (0, 255, 255), "building": (200, 140, 90), "wall": (120, 120, 180), "fence": (190, 153, 153),
    "barrier": (230, 90, 30), "pole": (153, 153, 153), "traffic light": (250, 170, 30), "traffic sign": (220, 220, 0),
    "bench": (130, 80, 180), "trash can": (60, 120, 60), "street object": (200, 0, 120), "person": (220, 20, 60),
    "rider": (255, 80, 80), "animal": (255, 170, 200), "car": (0, 0, 142), "bus": (0, 60, 100), "truck": (0, 0, 70),
    "motorcycle": (0, 0, 230), "bicycle": (119, 11, 32), "other vehicle": (0, 120, 200), "vegetation": (40, 140, 40),
    "sky": (135, 206, 235), "water": (0, 0, 160),
}
PAL = np.array([COLORS[n] for n in C.NAMES], np.uint8)


def adb(*args):
    r = subprocess.run(["adb", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stderr.strip()[-300:]}")
    return r.stdout


def run_chunk(model, frames, work, first):
    """Run one chunk on the phone; returns class maps (len, 128, 128) and the average NPU time in ms."""
    raw_dir = os.path.join(work, "in")
    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir)
    names = []
    for k, f in enumerate(frames):
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (CAMERA[1], CAMERA[0]):
            rgb = cv2.resize(rgb, CAMERA, interpolation=cv2.INTER_LINEAR)
        x = cv2.resize(rgb, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST).astype(np.float32)[None] / 255.0
        name = f"f{first + k:05d}.raw"
        x.tofile(os.path.join(raw_dir, name))
        names.append(name)
    adb("shell", f"rm -rf {REMOTE}/vid_in {REMOTE}/vid_out && mkdir -p {REMOTE}/vid_in")
    adb("push", raw_dir + "/.", f"{REMOTE}/vid_in/")
    adb("shell", f"cd {REMOTE} && ls vid_in/*.raw | sort | sed 's#^#{REMOTE}/#' > vid_list.txt")
    out = adb("shell", f"{ENV} && ./qnn-net-run --backend libQnnHtp.so --retrieve_context models/{model}.bin "
                       f"--input_list vid_list.txt --output_dir vid_out --perf_profile burst --profiling_level basic "
                       f"--use_native_output_files > vid.log 2>&1; echo rc=$?")
    if "rc=0" not in out:
        raise RuntimeError("qnn-net-run failed: " + adb("shell", f"tail -5 {REMOTE}/vid.log"))
    prof = adb("shell", f"{ENV} && ./qnn-profile-viewer --input_log vid_out/qnn-profiling-data_0.log")
    avg = re.search(r"Execute Stats \(Average\):(.*?)(?:Execute Stats|\Z)", prof, re.S).group(1)
    npu_ms = int(re.search(r"Accelerator \(execute\) time\): (\d+) us", avg).group(1)) / 1000
    local = os.path.join(work, "out")
    shutil.rmtree(local, ignore_errors=True)
    adb("pull", f"{REMOTE}/vid_out", local)
    maps = []
    for i in range(len(frames)):
        d = os.path.join(local, f"Result_{i}")
        f = os.path.join(d, os.listdir(d)[0])
        n = os.path.getsize(f)
        dtype = np.uint16 if n == OUT * OUT * NC * 2 else np.float32  # native (quantized) or float output
        maps.append(np.fromfile(f, dtype).reshape(OUT, OUT, NC).argmax(-1).astype(np.uint8))
    return maps, npu_ms


def legend(width):
    cols, row_h = 8, 22
    rows = (NC + cols - 1) // cols
    img = np.full((rows * row_h + 6, width, 3), 30, np.uint8)
    cw = width // cols
    for i, n in enumerate(C.NAMES):
        x, y = (i % cols) * cw + 6, (i // cols) * row_h + 4
        cv2.rectangle(img, (x, y), (x + 16, y + 14), tuple(int(v) for v in PAL[i][::-1]), -1)
        cv2.putText(img, n, (x + 22, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--chunk", type=int, default=300)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    print(f"{len(frames)} frames ({total} reported), {fps:.2f} fps, {frames[0].shape[1]}x{frames[0].shape[0]}")

    maps, times, t0 = [], [], time.time()
    for s in range(0, len(frames), a.chunk):
        m, ms = run_chunk(a.model, frames[s:s + a.chunk], a.work, s)
        maps += m
        times.append((ms, len(m)))
        print(f"  frames {s}-{s + len(m) - 1}: NPU {ms:.1f} ms/frame  ({time.time() - t0:.0f} s so far)")
    npu = sum(ms * n for ms, n in times) / sum(n for _, n in times)

    h, w = frames[0].shape[:2]
    leg = legend(2 * w)
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h + leg.shape[0]))
    counts = np.zeros(NC, np.int64)
    for f, m in zip(frames, maps):
        cls = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)  # stretch back like the app
        counts += np.bincount(cls.ravel(), minlength=NC)
        color = PAL[cls][..., ::-1]  # RGB -> BGR
        over = cv2.addWeighted(f, 0.45, color, 0.55, 0)
        cv2.putText(over, f"run3 w8a16 on phone NPU, {npu:.0f} ms/frame", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(np.vstack([np.hstack([f, over]), leg]))
    out.release()
    shutil.rmtree(a.work, ignore_errors=True)
    adb("shell", f"rm -rf {REMOTE}/vid_in {REMOTE}/vid_out")
    share = counts / counts.sum() * 100
    top = ", ".join(f"{C.NAMES[i]} {share[i]:.1f}%" for i in np.argsort(-share)[:8])
    print(f"\nNPU average {npu:.1f} ms/frame over {len(maps)} frames; total {time.time() - t0:.0f} s")
    print(f"Most common classes (share of all pixels): {top}")
    print(f"Saved {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
