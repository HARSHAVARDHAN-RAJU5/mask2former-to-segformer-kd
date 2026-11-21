"""Check a converted NPU model on the PC/server before the phone (run on nunnari, export venv + QAIRT env).

Runs the x86 model library with qnn-net-run (HTP backend = NPU numerics emulated on x86, or CPU backend),
then compares with ONNX Runtime (the float original) on phone-like validation photos (never used for
calibration):
  - class map match with ONNX (how often the argmax is the same)
  - mIoU against the human labels, for both the NPU model and ONNX

Usage:
  python npu_validate.py --mode fp16|w8a16 --work <segformer_b1_31 folder> --eval <eval_png dir> \
      --labels <eval_labels dir> [--n 40] [--backend libQnnHtp.so]
"""

import argparse
import glob
import os
import subprocess

import numpy as np
from PIL import Image

NAME, SIDE, OUT, NC = "segformer_b1_31", 512, 128, 31


def confusion(pred, label):
    v = label != 255
    return np.bincount(label[v].astype(np.int64) * NC + pred[v], minlength=NC * NC).reshape(NC, NC)


def miou(conf):
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    present = conf.sum(1) > 0
    return (inter[present] / np.maximum(union[present], 1)).mean()


def to_label_size(logits, shape):
    """(128,128,31) logits -> class map at the label's size (stretch back, nearest on the argmax)."""
    cls = logits.argmax(-1).astype(np.uint8)
    return np.asarray(Image.fromarray(cls).resize((shape[1], shape[0]), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--backend", default="libQnnHtp.so")
    a = ap.parse_args()
    sdk = os.environ["QNN_SDK_ROOT"] if "QNN_SDK_ROOT" in os.environ else os.environ["QAIRT_SDK_ROOT"]

    names = sorted(f[:-4] for f in os.listdir(a.eval) if f.endswith(".png"))[:a.n]
    run = os.path.join(a.work, "validate", a.mode)
    os.makedirs(os.path.join(run, "in"), exist_ok=True)
    xs = []
    for n in names:
        x = np.asarray(Image.open(os.path.join(a.eval, n + ".png")).convert("RGB"), dtype=np.float32)[None] / 255.0
        x.tofile(os.path.join(run, "in", n + ".raw"))
        xs.append(x)
    lst = os.path.join(run, "input_list.txt")
    with open(lst, "w") as f:
        f.write("\n".join(os.path.join(run, "in", n + ".raw") for n in names) + "\n")

    out = os.path.join(run, "out")
    subprocess.run(["rm", "-rf", out])
    so = os.path.join(a.work, "build", a.mode, "lib", "x86_64-linux-clang", f"lib{NAME}.so")
    r = subprocess.run(["qnn-net-run", "--backend", os.path.join(sdk, "lib", "x86_64-linux-clang", a.backend),
                        "--model", so, "--input_list", lst, "--output_dir", out], capture_output=True, text=True)
    with open(os.path.join(a.work, "logs", f"validate_{a.mode}.log"), "w") as f:
        f.write(r.stdout + r.stderr)
    res = sorted(glob.glob(os.path.join(out, "Result_*")), key=lambda d: int(d.rsplit("_", 1)[1]))
    if r.returncode != 0 or len(res) != len(names):
        print(f"qnn-net-run FAILED rc={r.returncode}: {(r.stdout + r.stderr).strip().splitlines()[-3:]}")
        return

    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(a.work, "onnx", NAME + ".onnx"), providers=["CPUExecutionProvider"])
    conf_npu, conf_onnx, match = np.zeros((NC, NC), np.int64), np.zeros((NC, NC), np.int64), []
    for n, x, d in zip(names, xs, res):
        ref = sess.run(None, {"image": x})[0][0]  # (128,128,31)
        got = np.fromfile(glob.glob(os.path.join(d, "*.raw"))[0], np.float32).reshape(OUT, OUT, NC)
        match.append((ref.argmax(-1) == got.argmax(-1)).mean())
        label = np.asarray(Image.open(os.path.join(a.labels, n + ".png")))
        conf_npu += confusion(to_label_size(got, label.shape), label)
        conf_onnx += confusion(to_label_size(ref, label.shape), label)
    print(f"RESULT {a.mode} ({a.backend}, {len(names)} val photos): class map match vs ONNX "
          f"{np.mean(match) * 100:.2f}% (worst {min(match) * 100:.2f}%) | mIoU NPU {miou(conf_npu):.4f} "
          f"vs ONNX {miou(conf_onnx):.4f}")


if __name__ == "__main__":
    main()
