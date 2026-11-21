# Decisions and tests

## Goal
Distil Mask2Former Swin-L (Mapillary Vistas v1.2) into SegFormer-B1 for outdoor segmentation on the phone.
Soft-label KD + human labels. Fallback: alpha = 0 (labels only).

## Models
| | Choice |
|---|---|
| Teacher | facebook/mask2former-swin-large-mapillary-vistas-semantic (offline, server only) |
| Student | SegFormer-B1 from nvidia/segformer-b1-finetuned-cityscapes-1024-1024, head = 31 classes (B0 if more speed needed) |
| Objects / indoor | Not this model: YOLOv8-seg and the indoor SegFormer-B0 cover them |

## Setup
| | Choice | Why |
|---|---|---|
| Server | advantech (RTX 5000 16 GB, 64 GB RAM) | nunnari: 11 GB GPU, 15 GB RAM, disk full |
| Python | Existing torch 2.11 / transformers 5.14, no venv | Already installed, GPU fp16 test passed |
| Long jobs | tmux | Survives SSH / laptop drops |
| Code | Written in this repo, then copied to the server | |

## Dataset
| | Result |
|---|---|
| Version | Mapillary Vistas v1.2 (teacher's 65 classes match it, same order) |
| Transfer | Server can't reach fbcdn.net; uploaded from PC with tools/upload_resume.py, checksum exact |
| Slim zip | Train/val images + v1.2 labels + config only: 19.5 GB instead of 29.3 GB |
| Check | 18,000 train + 2,000 val, every image has a label, values 0–65 |
| Note | Photo sizes vary (1024×768 up to 3680×2760) |

## Classes (kd/classes.py)
- 31 student classes; table in kd/classes.py.
- Ignore (255): Car Mount, Ego Vehicle, Unlabeled. Every other class is merged, never dropped.
- Matched by name. Check passed: 66/66 covered once, no unknown names.
- Teacher probabilities of merged classes are added up.
- Car Mount / Ego Vehicle teacher scores: left out, the rest rescaled to add up to 1.
- Ignore pixels: no CE loss and no KD loss.

## Teacher input size (kd/teacher_check.py, 5 photos)
| Size | ms/photo | GPU | Pixel acc | mIoU |
|---|---|---|---|---|
| 384×384 (squashed) | 64 | 2.0 GB | 93.6% | 45% |
| **Short side 1024, shape kept** | 375 | 3.7 GB | **97.4%** | **71%** |

Decision: 1024. Pictures clearly sharper (thin poles, barriers).
Load warning "swin.layernorm MISSING": layer not used; 97% match confirms the teacher works.

## Teacher cache (kd/teacher_cache.py)
- Per photo: top-5 classes + probabilities (fp16) at 1/4 of teacher input (e.g. 344×256), plus original size.
- Probabilities add up to 1 over 31 classes before keeping the top 5.
- Training photos only. Resumable (skips saved files), failed photos logged, not fatal.

| 100-photo test | Result |
|---|---|
| Time | 0.35 s/photo → ~1.7 h for 18,000 |
| Size | 1.37 MB/photo → ~24.6 GB |
| Top-5 hold | 99.72% of the probability (worst photo 97.55%) |
| Re-load check | 100% same top class, 0 difference |
| Resume test | Restarted at 95/100, redid only the 5 missing |
| Broken photo test | Logged and skipped, job finished |

| Full run (18,000 photos, tmux) | Result |
|---|---|
| Saved / failed | 18,000 / 0 (17,900 new in 103 min + 100 from the test) |
| Time | 0.35 s/photo |
| Size | 24 GB (1.38 MB/photo) |
| Top-5 hold | 99.72% on average; worst photo 88.05% |
| Checks | Every photo has a file and none extra; 200 random files open OK (5 classes, ids < 31, probs valid); 3 re-loads match the live teacher exactly |

## Phone input: 512×512
- Deployed SegFormer input is fixed at 1×512×512×3 RGB (values 0–1), output 128×128.
- The app **stretches** the whole camera frame (640×480) to 512×512, nearest-neighbour, no crop / pad
  (`engine/core/semseg.cpp` in the app).
- So the student is trained and scored at that scale and shape: every frame is stretched to a square.
- Photos wider than 4:3 (12.6% are 16:9, 2% other) are first cut to 4:3, like the camera:
  training = random horizontal position, validation = centred. Same cut on photo, label and teacher map.
  4:3 and narrower photos are kept whole.
- Validation photo = centred 4:3 cut → camera size 640×480 → **nearest-neighbour** stretch to 512×512 (the app's step).
  Training keeps bilinear. *Revisit if phone results look worse than validation.*

## Training (kd/dataset.py, kd/train.py)
- Teacher map: top 5 → full 31-class map (rest 0, rescaled to 1), then the SAME 4:3 cut / stretch / zoom / crop / flip as photo and label
  (exact fractional cut positions for all three).
  Teacher cache kept as is (not regenerated): the rebuilt map is resized with the photo.
- Map is always exactly 1/4 of the photo; cuts start at multiples of 4 (map pixel = 4×4 photo pixels).
- Stretch to a square of side 512 × random zoom 0.5–1.5, then cut 512×512; flip 50%; colour change on the photo only.
  Padding (when zoomed below 512) gets label 255.
- Student output is 1/4 of the input, same as the map: KD needs no resizing; CE compares at full crop size.

| Setting | Choice | Why |
|---|---|---|
| KD formula | T² · KL(teacher_T ‖ student_T), teacher_T = p^(1/T) rescaled (= softmax(log p / T)) | Cache holds probabilities, not raw scores |
| KD / CE masking | KD: valid cells are **selected** before the maths (not multiplied by 0); never log(0), classes outside the top 5 stay exactly 0. CE: all-ignore batch → 0, not NaN | Padded teacher cells are all zeros; NaN must never leak. Tested: no NaN, padded cells get 0 KD gradient, same numbers as before |
| T | 2 | Task file |
| alpha | 0.5, plus a baseline run with 0 | Baseline shows whether the teacher helped |
| Crop / batch | 512 × 512, real batch 8, no accumulation | Match the phone input; fits easily (was 768, batch 4 × 2) |
| Optimizer | AdamW 6e-5, new last layer ×10, weight decay 0.01, warm-up 1500, straight-line decay | SegFormer standard |
| Length | Decide after measuring speed (default 40,000 steps) | |
| Validation | Every 4,000 steps, phone-like photo (see above), answer stretched back and scored at the label's full size (centred 4:3 cut); keep best + last + per-class IoU | Same as the phone |
| Runs | alpha 0.5 (runs/kd_a0.5), then alpha 0 baseline (runs/kd_a0), back-to-back in one tmux session | Baseline shows whether the teacher helped |
| Resume | Restart continues from last.pt (saved every 1,000 steps) | |

| Test (512, stretched, batch 8) | Result |
|---|---|
| Line-up (6 cuts) | Teacher top class = label on 96–99% of pixels; pictures line up |
| 50-step run | Loss 7.62 → 3.02 (CE 3.18 → 0.98, KD 12.1 → 5.1); val mIoU 0.101 on 20 photos (expected this early) |
| Speed / memory | 0.22 s/step → 40,000 steps ≈ 2.5 h + validation; peak GPU 3.1 GB of 16 |
| Resume | Tested at 768 (restart at step 50 → 70, loss kept falling); same code path |
| 4:3 cut line-up (6 wide photos, 16:9) | Teacher top class = label on 93–97% of pixels; pictures line up (incl. zoomed-out + flipped) |
| 4:3 cut validation | 16:9 val photos → label cut to exactly 1.333:1; photo 512×512 |
| 10-step run (4:3 cut) | Loss 7.88 → 7.24 (warm-up), GPU 3.1 GB; val on 50 photos takes 3 s (→ ~2 min for 2,000) |

## Export (later)
- Build the colour normalization (ImageNet mean/std) into the exported model, so the phone keeps feeding RGB 0–1.
