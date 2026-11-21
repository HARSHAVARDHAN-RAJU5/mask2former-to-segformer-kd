"""Train the student (SegFormer-B1, 31 classes) with soft labels from the teacher cache + human labels.

loss = alpha * KD + (1 - alpha) * CE
  CE = cross-entropy between the student and the human label (ignore pixels skipped)
  KD = T^2 * KL(teacher_T || student_T) per pixel, averaged over non-ignore pixels
       teacher_T = p^(1/T) rescaled to add up to 1  (= softmax(log p / T): the cache holds
                   probabilities, not raw scores)
       student_T = softmax(student scores / T)
alpha = 0 -> plain supervised baseline (KD not computed).

Modes:
  --lineup N      save N pictures (photo | label | teacher map) after 4:3 cut/stretch/zoom/crop/flip, then exit
                  (add --lineup-wide to check only photos wider than 4:3)
  (default)       train; resumes from <out>/last.pt if it exists
  --max-steps N   stop after N steps in this session (test runs)

Usage (on the server, inside tmux):
  python3 kd/train.py --data <mapillary_vistas> --cache <teacher_cache> --out <run folder> [options]
"""

import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("HF_HOME", "/mnt/nunnari3/hf_cache")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import classes as C  # noqa: E402
from kd.dataset import MEAN, STD, IndianSet, TrainSet, ValSet, frame_confidences  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import SegformerForSemanticSegmentation  # noqa: E402

STUDENT = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"


def kd_ce_loss(logits, y, t, alpha, T, kd_mask=None):
    """logits (B,31,h,w) student scores at 1/4 size; y (B,H,W) labels; t (B,31,h,w) teacher probs;
    kd_mask (B,h,w) bool = cells to learn from the teacher (default: label present at the cell centre)."""
    logits = logits.float()
    up = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    if (y != C.IGNORE_ID).any():
        ce = F.cross_entropy(up, y, ignore_index=C.IGNORE_ID)
    else:  # every pixel ignored: 0 instead of 0/0 = NaN
        ce = logits.sum() * 0.0
    if alpha == 0:
        return ce, ce.detach(), torch.zeros((), device=ce.device)
    valid = kd_mask if kd_mask is not None else y[:, 2::4, 2::4] != C.IGNORE_ID
    # select valid cells first (padding / ignore cells never enter the maths), then (cells, 31)
    tv = t.float().permute(0, 2, 3, 1)[valid]
    lv = logits.permute(0, 2, 3, 1)[valid]
    if tv.shape[0] == 0:
        kd = logits.sum() * 0.0
    else:
        # = softmax(log(p) / T) without ever taking log(0): classes outside the top 5 stay exactly 0
        tgt = tv.pow(1.0 / T)
        tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-8)
        logp = F.log_softmax(lv / T, dim=1)
        kd = (tgt * (tgt.clamp_min(1e-8).log() - logp)).sum(1).mean() * T * T  # 0 * finite = 0
    return alpha * kd + (1 - alpha) * ce, ce.detach(), kd.detach()


def lr_factor(step, iters, warmup):
    """Linear warm-up, then straight-line decay to 0 at the last step."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, 1 - (step - warmup) / max(1, iters - warmup))


@torch.no_grad()
def evaluate(model, loader):
    """mIoU over the classes present in the human labels. The student's square answer is stretched
    back to the label's full size and scored there."""
    model.eval()
    n = C.NUM_CLASSES
    conf = torch.zeros(n * n, dtype=torch.int64, device="cuda")
    for x, y in tqdm(loader, desc="val", leave=False, mininterval=5):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(pixel_values=x).logits
        pred = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        v = y != C.IGNORE_ID
        conf += torch.bincount(y[v] * n + pred[v], minlength=n * n)
    model.train()
    conf = conf.view(n, n).double().cpu()
    inter = conf.diag()
    iou = inter / (conf.sum(0) + conf.sum(1) - inter).clamp_min(1)
    present = conf.sum(1) > 0
    per_class = {C.NAMES[k]: (round(iou[k].item(), 4) if present[k] else None) for k in range(n)}
    return iou[present].mean().item(), (inter.sum() / conf.sum()).item(), per_class


def palette():
    rng = np.random.RandomState(3)  # same colours as teacher_check.py
    p = rng.randint(40, 255, size=(256, 3)).astype(np.uint8)
    p[C.IGNORE_ID] = 0
    return p


def lineup(ds, n, out, wide_only=False):
    """Pictures of photo | label | teacher top class after 4:3 cut/stretch/zoom/crop/flip, plus a number:
    how often the teacher's top class equals the label (should be high if they line up).
    wide_only: pick only photos wider than 4:3 (the ones that get the 4:3 cut)."""
    pal = palette()
    order = random.Random(0).sample(range(len(ds)), len(ds))
    if wide_only:
        from kd.dataset import cut_4_3
        order = [i for i in order[:3000]
                 if cut_4_3(*Image.open(os.path.join(ds.img_dir, ds.names[i] + ".jpg")).size, False)[1] < 1]
    for k, i in enumerate(order[:n]):
        x, y, t, m = ds[i]
        photo = ((x * STD + MEAN).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        lab = pal[y.numpy().astype(np.uint8)]
        top = t.float().argmax(0).numpy().astype(np.uint8)
        tea = np.repeat(np.repeat(pal[top], 4, 0), 4, 1)
        keep = np.repeat(np.repeat(m.numpy(), 4, 0), 4, 1)[..., None]
        learned = np.where(keep, photo, photo // 4)  # dark = the teacher is NOT learned there
        yc = y[2::4, 2::4].numpy()
        v = yc != C.IGNORE_ID
        agree = (top[v] == yc[v]).mean() * 100 if v.any() else float("nan")
        Image.fromarray(np.concatenate([photo, lab, tea, learned], 1)).save(os.path.join(out, f"lineup_{k}.jpg"), quality=90)
        w, h = Image.open(os.path.join(ds.img_dir, ds.names[i] + ".jpg")).size
        print(f"  lineup_{k}.jpg  {ds.names[i]}  ({w}x{h}, {w / h:.2f}:1)  teacher top class = label on "
              f"{agree:.1f}% of labelled pixels; teacher learned on {m.float().mean() * 100:.0f}% of cells")


def indian_names(a):
    """Indian frames to train on: in the keep list (indoor/ATM already removed), cached, and with average
    teacher confidence >= --indian-frame-conf. Confidences are computed once and stored in the run folder."""
    listed = [l.strip()[:-4] for l in open(a.indian_list) if l.strip()]
    names = [n for n in listed if os.path.exists(os.path.join(a.indian_cache, n + ".npz"))]
    if len(names) < len(listed):
        print(f"WARNING: only {len(names)} of {len(listed)} listed Indian frames are cached")
    path = os.path.join(a.out, "indian_frame_conf.json")
    conf = json.load(open(path)) if os.path.exists(path) else {}
    todo = [n for n in names if n not in conf]
    if todo:
        conf.update(frame_confidences(a.indian_cache, todo))
        with open(path, "w") as f:
            json.dump(conf, f)
    keep = [n for n in names if conf[n] >= a.indian_frame_conf]
    print(f"Indian frames: {len(names)} cached, {len(keep)} with average teacher confidence >= {a.indian_frame_conf}")
    return keep


def save(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--side", type=int, default=512, help="phone input: frame stretched to side x side")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--iters", type=int, default=40000, help="total optimizer steps")
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--head-lr-mult", type=float, default=10.0)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--val-every", type=int, default=4000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-limit", type=int, default=0, help="only the first N val photos (0 = all 2000)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps this session (0 = no limit)")
    ap.add_argument("--lineup", type=int, default=0)
    ap.add_argument("--lineup-wide", action="store_true", help="line-up check on photos wider than 4:3 only")
    ap.add_argument("--lineup-indian", action="store_true", help="line-up check on the Indian frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="", help="start from a COPY of these weights (e.g. a saved best.pt); "
                                               "the file itself is only read")
    ap.add_argument("--indian-images", default="", help="unlabeled Indian frames (teacher only)")
    ap.add_argument("--indian-cache", default="")
    ap.add_argument("--indian-list", default="", help="frame file names to use (indoor/ATM already removed)")
    ap.add_argument("--indian-frame-conf", type=float, default=0.80, help="drop frames with lower average confidence")
    ap.add_argument("--indian-pixel-conf", type=float, default=0.90, help="learn the teacher only where this sure")
    ap.add_argument("--indian-share", type=float, default=0.5, help="share of each batch that is Indian frames")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.init:
        assert os.path.dirname(os.path.abspath(a.init)) != os.path.abspath(a.out), \
            "--out must not be the folder of --init (the saved model must never be written)"

    train_ds = TrainSet(a.data, a.cache, side=a.side, crop=a.crop)
    indian_ds = None
    if a.indian_images:
        indian_ds = IndianSet(a.indian_images, a.indian_cache, indian_names(a), side=a.side, crop=a.crop,
                              pixel_conf=a.indian_pixel_conf)
    if a.lineup:
        lineup(indian_ds if a.lineup_indian else train_ds, a.lineup, a.out, wide_only=a.lineup_wide)
        return

    model = SegformerForSemanticSegmentation.from_pretrained(
        STUDENT, num_labels=C.NUM_CLASSES, id2label=dict(enumerate(C.NAMES)),
        label2id={n: i for i, n in enumerate(C.NAMES)}, ignore_mismatched_sizes=True).cuda().train()
    is_head = lambda n: n.startswith("decode_head.classifier")  # noqa: E731  new 31-class layer
    groups = [{"params": [p for n, p in model.named_parameters() if not is_head(n)], "lr": a.lr},
              {"params": [p for n, p in model.named_parameters() if is_head(n)], "lr": a.lr * a.head_lr_mult}]
    base_lrs = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda")

    step, best = 0, -1.0
    last_path, best_path = os.path.join(a.out, "last.pt"), os.path.join(a.out, "best.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        print(f"Resuming from step {step} (best val mIoU so far {best:.4f})")
    elif a.init:
        model.load_state_dict(torch.load(a.init, map_location="cuda", weights_only=False)["model"])
        print(f"Starting from a copy of {a.init}")
    else:
        print(f"Starting from {STUDENT}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    def endless(ds, batch, workers):
        dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True,
                                         num_workers=workers, pin_memory=True, persistent_workers=True)
        while True:
            yield from dl

    if indian_ds is None:
        it = endless(train_ds, a.batch, a.workers)
    else:  # every batch = Mapillary part + Indian part, joined
        b_ind = round(a.batch * a.indian_share)
        print(f"Each batch: {a.batch - b_ind} Mapillary + {b_ind} Indian")
        it_map = endless(train_ds, a.batch - b_ind, max(1, a.workers // 2))
        it_ind = endless(indian_ds, b_ind, max(1, a.workers // 2))
        it = (tuple(torch.cat(p) for p in zip(next(it_map), next(it_ind))) for _ in iter(int, 1))
    val_loader = torch.utils.data.DataLoader(ValSet(a.data, side=a.side, limit=a.val_limit), batch_size=1, num_workers=4)

    stop_at = min(a.iters, step + a.max_steps) if a.max_steps else a.iters
    train_log = open(os.path.join(a.out, "train_log.csv"), "a")
    val_log = open(os.path.join(a.out, "val_log.csv"), "a")
    bar = tqdm(total=a.iters, initial=step, desc="train", unit="step", dynamic_ncols=True, mininterval=2)
    torch.cuda.reset_peak_memory_stats()
    run = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "n": 0}
    start_step, train_time, t_last = step, 0.0, time.time()

    while step < stop_at:
        f = lr_factor(step, a.iters, a.warmup)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * f
        for _ in range(a.accum):
            x, y, t, m = (v.cuda(non_blocking=True) for v in next(it))
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(pixel_values=x).logits
            loss, ce, kd = kd_ce_loss(logits, y, t, a.alpha, a.T, kd_mask=m)
            scaler.scale(loss / a.accum).backward()
            run["loss"] += loss.item()
            run["ce"] += ce.item()
            run["kd"] += kd.item()
            run["n"] += 1
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        bar.update(1)

        if step % a.log_every == 0 or step == stop_at:
            now = time.time()
            train_time += now - t_last
            t_last = now
            m = {k: run[k] / max(1, run["n"]) for k in ("loss", "ce", "kd")}
            if not math.isfinite(m["loss"]):
                bar.write(f"WARNING step {step}: loss is {m['loss']}")
            sec = train_time / (step - start_step)
            mem = torch.cuda.max_memory_allocated() / 2**30
            train_log.write(f"{step},{m['loss']:.4f},{m['ce']:.4f},{m['kd']:.4f},"
                            f"{opt.param_groups[0]['lr']:.3e},{sec:.3f},{mem:.2f}\n")
            train_log.flush()
            bar.set_postfix(loss=f"{m['loss']:.3f}", ce=f"{m['ce']:.3f}", kd=f"{m['kd']:.3f}")
            run = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "n": 0}

        do_val = step % a.val_every == 0 or step == a.iters
        if do_val:
            t_val = time.time()
            miou, acc, per_class = evaluate(model, val_loader)
            val_log.write(f"{step},{miou:.4f},{acc:.4f}\n")
            val_log.flush()
            bar.write(f"[val] step {step}: mIoU {miou:.4f}, pixel acc {acc:.4f}, {time.time() - t_val:.0f} s"
                      + ("  (new best)" if miou > best else ""))
            if miou > best:
                best = miou
                save(best_path, {"model": model.state_dict(), "step": step, "miou": miou, "classes": C.NAMES})
                with open(os.path.join(a.out, "best_iou.json"), "w") as fj:  # per-class IoU of the best model
                    json.dump({"step": step, "miou": miou, "pixel_acc": acc,
                               "train_hours_so_far": round(train_time / 3600, 2), "per_class_iou": per_class}, fj, indent=1)
            t_last += time.time() - t_val  # validation time not counted as training time
        if do_val or step % a.save_every == 0 or step == stop_at:
            save(last_path, {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                             "step": step, "best": best})
    bar.close()

    done = step - start_step
    if done:
        sec = train_time / done
        print(f"\nThis session: {done} steps, {sec:.2f} s/step, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        print(f"  -> {a.iters} steps would take {sec * a.iters / 3600:.1f} h (without validation)")
    print(f"Stopped at step {step} / {a.iters}; best val mIoU {best:.4f}")


if __name__ == "__main__":
    main()
