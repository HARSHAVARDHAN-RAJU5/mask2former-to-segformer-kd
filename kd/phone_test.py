"""Test NPU models on the phone over adb (run on the PC with the phone connected).

  speed:    each model runs the same input N times (qnn-net-run, burst); NPU time read from the profile log
  accuracy: phone-like Mapillary val photos through our 31-class builds; the phone's answers are compared with
            ONNX Runtime on the PC (class-map match) and with the human labels (mIoU)

Usage:
  python kd/phone_test.py --models b0_fp16 b0_w8a16 b1_fp16 b1_w8a16 --acc b1_fp16 b1_w8a16 \
      --eval <eval png dir> --labels <eval label dir> --onnx <segformer_b1_31.onnx> --work <PC work dir>
Models must already be on the phone as <REMOTE>/models/<name>.bin, with the QAIRT runtime in <REMOTE>.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess

import numpy as np
from PIL import Image

REMOTE = "/data/local/tmp/sfb1"
ENV = f"cd {REMOTE} && export LD_LIBRARY_PATH={REMOTE} ADSP_LIBRARY_PATH={REMOTE}"
NC, OUT = 31, 128


def adb(*args, check=True):
    r = subprocess.run(["adb", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stderr.strip()[-300:]}")
    return r.stdout


def net_run(model, input_list, out_dir, profile=False):
    extra = " --profiling_level basic" if profile else ""
    out = adb("shell", f"{ENV} && rm -rf {out_dir} && ./qnn-net-run --backend libQnnHtp.so "
              f"--retrieve_context models/{model}.bin --input_list {input_list} --output_dir {out_dir} "
              f"--perf_profile burst{extra} > {out_dir}.log 2>&1; echo rc=$?")
    if "rc=0" not in out:
        raise RuntimeError(f"{model}: qnn-net-run failed: " + adb("shell", f"tail -5 {REMOTE}/{out_dir}.log"))


def speed(model, n):
    adb("shell", f"cd {REMOTE} && for i in $(seq 1 {n}); do echo {REMOTE}/in/speed_input.raw; done > speed_list.txt")
    net_run(model, "speed_list.txt", "out_speed", profile=True)
    text = adb("shell", f"{ENV} && ./qnn-profile-viewer --input_log out_speed/qnn-profiling-data_0.log")
    stats = {}
    for block in ("Average", "Min", "Max"):
        m = re.search(rf"Execute Stats \({block}\):(.*?)(?:Execute Stats|\Z)", text, re.S)
        acc = re.search(r"Backend \(Accelerator \(execute\) time\): (\d+) us", m.group(1))
        netrun = re.search(r"NetRun: (\d+) us", m.group(1))
        stats[block] = (int(acc.group(1)) / 1000, int(netrun.group(1)) / 1000)
    load = re.search(r"Init Stats:.*?NetRun: (\d+) us", text, re.S)
    ips = re.search(r"NetRun IPS \(includes IO and misc. time\): ([\d.]+)", text)
    return {"npu_avg": stats["Average"][0], "npu_min": stats["Min"][0], "npu_max": stats["Max"][0],
            "netrun_avg": stats["Average"][1], "load_ms": int(load.group(1)) / 1000, "ips": float(ips.group(1))}


def m2f_scores(result_dir, matrix):
    """Mask2Former phone outputs -> our 31-class scores (128,128,31), same maths as the teacher cache:
    class probabilities (drop "no object") x mask probabilities, summed over the 100 queries, then merged."""
    raws = glob.glob(os.path.join(result_dir, "*.raw"))
    q = np.fromfile(next(p for p in raws if "class_queries" in p), np.float32).reshape(100, -1)
    mk = np.fromfile(next(p for p in raws if "masks_queries" in p), np.float32).reshape(OUT, OUT, 100)
    e = np.exp(q - q.max(-1, keepdims=True))
    cls = (e / e.sum(-1, keepdims=True))[:, :-1]  # (100, 65)
    per_pixel = np.einsum("qc,hwq->hwc", cls, 1.0 / (1.0 + np.exp(-mk)))  # (128, 128, 65)
    return per_pixel @ matrix  # (128, 128, 31); argmax is all we need


def confusion(pred, label):
    v = label != 255
    return np.bincount(label[v].astype(np.int64) * NC + pred[v], minlength=NC * NC).reshape(NC, NC)


def miou(conf):
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    present = conf.sum(1) > 0
    return (inter[present] / np.maximum(union[present], 1)).mean()


def to_label_size(logits, shape):
    cls = logits.argmax(-1).astype(np.uint8)
    return np.asarray(Image.fromarray(cls).resize((shape[1], shape[0]), Image.NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--acc", nargs="*", default=[])
    ap.add_argument("--acc-m2f", nargs="*", default=[], help="Mask2Former models (100 query outputs) to score")
    ap.add_argument("--config", default="", help="config_v1.2.json (teacher class order) for --acc-m2f")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-speed", type=int, default=100)
    ap.add_argument("--n-acc", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    print("== speed (same frame, %d runs each, burst)" % a.n_speed)
    print(f"{'model':10s} {'NPU avg':>8s} {'min':>6s} {'max':>6s} {'net-run avg':>11s} {'IPS':>6s} {'load':>7s}")
    for m in a.models:
        s = speed(m, a.n_speed)
        print(f"{m:10s} {s['npu_avg']:6.1f}ms {s['npu_min']:6.1f} {s['npu_max']:6.1f} {s['netrun_avg']:9.1f}ms "
              f"{s['ips']:6.1f} {s['load_ms']:5.0f}ms")
    if not a.acc and not a.acc_m2f:
        return

    names = sorted(f[:-4] for f in os.listdir(a.eval) if f.endswith(".png"))[:a.n_acc]
    raw_dir = os.path.join(a.work, "eval_raw")
    os.makedirs(raw_dir, exist_ok=True)
    xs = []
    for n in names:
        x = np.asarray(Image.open(os.path.join(a.eval, n + ".png")).convert("RGB"), dtype=np.float32)[None] / 255.0
        x.tofile(os.path.join(raw_dir, n + ".raw"))
        xs.append(x)
    adb("shell", f"rm -rf {REMOTE}/eval && mkdir -p {REMOTE}/eval")
    adb("push", raw_dir + "/.", f"{REMOTE}/eval/")
    adb("shell", f"cd {REMOTE} && ls eval/*.raw | sort | sed 's#^#{REMOTE}/#' > eval_list.txt")
    listed = adb("shell", f"cat {REMOTE}/eval_list.txt").split()
    assert [os.path.basename(p)[:-4] for p in listed] == names, "phone input order differs from PC order"

    import onnxruntime as ort
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    refs = [sess.run(None, {"image": x})[0][0] for x in xs]
    labels = [np.asarray(Image.open(os.path.join(a.labels, n + ".png"))) for n in names]
    conf_ref = sum(confusion(to_label_size(r, l.shape), l) for r, l in zip(refs, labels))

    print(f"\n== accuracy ({len(names)} Mapillary val photos, phone-like, never used for calibration)")
    print(f"{'model':10s} {'match vs ONNX':>14s} {'worst':>7s} {'mIoU phone':>11s} {'mIoU ONNX':>10s}")
    for m in a.acc:
        net_run(m, "eval_list.txt", f"out_{m}")
        local = os.path.join(a.work, f"out_{m}")
        shutil.rmtree(local, ignore_errors=True)
        adb("pull", f"{REMOTE}/out_{m}", local)
        conf, match = np.zeros((NC, NC), np.int64), []
        for i, (r, l) in enumerate(zip(refs, labels)):
            f = glob.glob(os.path.join(local, f"Result_{i}", "*.raw"))
            got = np.fromfile(f[0], np.float32).reshape(OUT, OUT, NC)
            match.append((got.argmax(-1) == r.argmax(-1)).mean())
            conf += confusion(to_label_size(got, l.shape), l)
        print(f"{m:10s} {np.mean(match) * 100:13.2f}% {min(match) * 100:6.1f}% {miou(conf):11.4f} {miou(conf_ref):10.4f}")
        shutil.rmtree(local, ignore_errors=True)
        adb("shell", f"rm -rf {REMOTE}/out_{m}")

    if a.acc_m2f:
        import json
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from kd import classes as C
        names65 = [l["readable"] for l in json.load(open(a.config))["labels"]][:65]
        matrix = C.teacher_merge_matrix(names65).numpy()
        for m in a.acc_m2f:
            net_run(m, "eval_list.txt", f"out_{m}")
            local = os.path.join(a.work, f"out_{m}")
            shutil.rmtree(local, ignore_errors=True)
            adb("pull", f"{REMOTE}/out_{m}", local)
            conf = np.zeros((NC, NC), np.int64)
            for i, l in enumerate(labels):
                conf += confusion(to_label_size(m2f_scores(os.path.join(local, f"Result_{i}"), matrix), l.shape), l)
            print(f"{m:10s} {'(65 -> 31)':>14s} {'':>7s} {miou(conf):11.4f}")
            shutil.rmtree(local, ignore_errors=True)
            adb("shell", f"rm -rf {REMOTE}/out_{m}")


if __name__ == "__main__":
    main()
