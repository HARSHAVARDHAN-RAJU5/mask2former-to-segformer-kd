"""Train the student (SegFormer-B1, 31 classes) with soft labels from the teacher cache + human labels.

loss = alpha * KD + (1 - alpha) * CE
  CE = cross-entropy between the student and the human label (ignore pixels skipped)
  KD = T^2 * KL(teacher_T || student_T) per pixel, averaged over non-ignore pixels
       teacher_T = p^(1/T) rescaled to add up to 1  (= softmax(log p / T): the cache holds
                   probabilities, not raw scores)
       student_T = softmax(student scores / T)
alpha = 0 -> plain supervised baseline (KD not computed).

Modes:
  --lineup N      save N pictures (photo | label | teacher map) after zoom/cut/flip, then exit
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
from kd.dataset import MEAN, STD, TrainSet, ValSet  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import SegformerForSemanticSegmentation  # noqa: E402

STUDENT = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"


def kd_ce_loss(logits, y, t, alpha, T):
    """logits (B,31,h,w) student scores at 1/4 size; y (B,H,W) labels; t (B,31,h,w) teacher probs."""
    logits = logits.float()
    up = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
    ce = F.cross_entropy(up, y, ignore_index=C.IGNORE_ID)
    if alpha == 0:
        return ce, ce.detach(), torch.zeros((), device=ce.device)
    valid = (y[:, 2::4, 2::4] != C.IGNORE_ID).float()  # label at the centre of each 4x4 block
    tgt = t.float().pow(1.0 / T)
    tgt = tgt / tgt.sum(1, keepdim=True).clamp_min(1e-8)
    logp = F.log_softmax(logits / T, dim=1)
    kl = (tgt * (tgt.clamp_min(1e-8).log() - logp)).sum(1)  # (B,h,w); 0 where tgt is 0
    kd = (kl * valid).sum() / valid.sum().clamp_min(1.0) * T * T
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
    return iou[present].mean().item(), (inter.sum() / conf.sum()).item(), iou


def palette():
    rng = np.random.RandomState(3)  # same colours as teacher_check.py
    p = rng.randint(40, 255, size=(256, 3)).astype(np.uint8)
    p[C.IGNORE_ID] = 0
    return p


def lineup(ds, n, out):
    """Pictures of photo | label | teacher top class after zoom/cut/flip, plus a number:
    how often the teacher's top class equals the label (should be high if they line up)."""
    pal = palette()
    for k, i in enumerate(random.Random(0).sample(range(len(ds)), n)):
        x, y, t = ds[i]
        photo = ((x * STD + MEAN).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        lab = pal[y.numpy().astype(np.uint8)]
        top = t.float().argmax(0).numpy().astype(np.uint8)
        tea = np.repeat(np.repeat(pal[top], 4, 0), 4, 1)
        yc = y[2::4, 2::4].numpy()
        v = yc != C.IGNORE_ID
        agree = (top[v] == yc[v]).mean() * 100 if v.any() else float("nan")
        Image.fromarray(np.concatenate([photo, lab, tea], 1)).save(os.path.join(out, f"lineup_{k}.jpg"), quality=90)
        print(f"  lineup_{k}.jpg  {ds.names[i]}  teacher top class = label on {agree:.1f}% of labelled pixels")


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
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    train_ds = TrainSet(a.data, a.cache, side=a.side, crop=a.crop)
    if a.lineup:
        lineup(train_ds, a.lineup, a.out)
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
    else:
        print(f"Starting from {STUDENT}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch, shuffle=True, drop_last=True,
                                         num_workers=a.workers, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(ValSet(a.data, side=a.side, limit=a.val_limit), batch_size=1, num_workers=4)

    def batches():
        while True:
            yield from loader
    it = batches()

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
            x, y, t = next(it)
            x, y, t = x.cuda(non_blocking=True), y.cuda(non_blocking=True), t.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(pixel_values=x).logits
            loss, ce, kd = kd_ce_loss(logits, y, t, a.alpha, a.T)
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
            miou, acc, _ = evaluate(model, val_loader)
            val_log.write(f"{step},{miou:.4f},{acc:.4f}\n")
            val_log.flush()
            bar.write(f"[val] step {step}: mIoU {miou:.4f}, pixel acc {acc:.4f}"
                      + ("  (new best)" if miou > best else ""))
            if miou > best:
                best = miou
                save(best_path, {"model": model.state_dict(), "step": step, "miou": miou, "classes": C.NAMES})
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
