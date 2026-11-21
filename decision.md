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

## Results (Mapillary val, 2,000 photos, phone-like 512×512)
| Run | Best mIoU | Pixel acc | Best at step | Training time |
|---|---|---|---|---|
| alpha 0.5 (KD + CE) | 0.5354 | 0.9262 | 40,000 (last; still rising slightly) | 1 h 56 min + 10 × ~93 s validation |
| alpha 0 (CE only, baseline) | **0.5371** | 0.9267 | 40,000 (last; still rising slightly) | 1 h 56 min + 10 × ~93 s validation |

**Result: KD did not help on Mapillary val** (−0.002 mIoU, within noise; at every check the two runs were within ~0.005 after step 8k).
Per-class differences are small and mixed. Both runs: pothole and animal 0.00, curb cut 0.09 (rare classes not learned).

| Class | KD | Base | | Class | KD | Base | | Class | KD | Base |
|---|---|---|---|---|---|---|---|---|---|---|
| road | 0.915 | 0.915 | | wall | 0.489 | 0.481 | | rider | 0.526 | 0.543 |
| sidewalk | 0.751 | 0.747 | | fence | 0.555 | 0.551 | | animal | 0.000 | 0.000 |
| crosswalk | 0.552 | 0.548 | | barrier | 0.549 | 0.545 | | car | 0.883 | 0.885 |
| curb | 0.508 | 0.515 | | pole | 0.452 | 0.480 | | bus | 0.721 | 0.716 |
| curb cut | 0.092 | 0.088 | | traffic light | 0.533 | 0.544 | | truck | 0.670 | 0.673 |
| terrain | 0.674 | 0.675 | | traffic sign | 0.587 | 0.594 | | motorcycle | 0.496 | 0.498 |
| rail track | 0.459 | 0.485 | | bench | 0.155 | 0.138 | | bicycle | 0.525 | 0.534 |
| pothole | 0.000 | 0.000 | | trash can | 0.354 | 0.345 | | other vehicle | 0.229 | 0.236 |
| manhole | 0.363 | 0.372 | | street object | 0.406 | 0.403 | | vegetation | 0.877 | 0.879 |
| building | 0.851 | 0.853 | | person | 0.644 | 0.648 | | sky | 0.977 | 0.978 |
| | | | | | | | | water | 0.805 | 0.782 |

**Saved models** (copies, never trained on): `saved_models/mapillary_kd_a0.5/` and `saved_models/mapillary_kd_a0/`
(best.pt + best_iou.json + args + val log; checksums match the run folders).
**Rule: any further training starts from a COPY of a saved model, never overwrites it.**

## Indian frames (next stage)
- Source: Haidata YOLO delivery, 36,365 unique frames from 1,157 videos (~31 frames each), ~15.8 GB.
  Indoor and outdoor mixed (classes include Door, Elevator, Couch as well as Pothole, Open drainage, Cow).
- Only the images are needed: the teacher makes the labels; the YOLO boxes are not used.
- Concern: noisy frames (blur, dark, indoor, near-duplicates) could teach the student wrong answers.

**Visual test, 24 frames** (kd/compare_frames.py; 4 folders × 6, 24 different videos, all 16:9 → centred 4:3 cut,
phone-like 512×512; all answers converted to the app's 15 navigation classes; no human labels, so the teacher is the
reference, not the truth — biased in favour of our students, which learned from Mapillary / the teacher):

| | KD student | Base student | B1 Cityscapes | B0 ADE (phone) |
|---|---|---|---|---|
| Agreement with teacher (nav classes) | 78.0% | 77.1% | 63.8% | 67.9% |
| Walkable area IoU vs teacher | 86.9% | 86.2% | 76.6% | 75.8% |

- On clear street frames our students look clearly better than B0 and B1 Cityscapes (shops, people, footpaths).
  KD vs base: no visible difference (same as on Mapillary).
- Teacher confidence flags bad frames: average 0.88, 81% of pixels ≥ 70% sure. Worst: an indoor ATM glass booth
  (0.50, 0% sure; teacher called it "vehicle"). 3 of 24 frames below 0.80 — all odd views (indoor, close-up
  looking down, drain).
- No model has Indian classes (open drain, speed breaker, …): the teacher maps them to the nearest Mapillary class.
- Head-camera views looking down at the ground are hard for every model (Mapillary is car-mounted, looking forward).
- B0 here = `nvidia/segformer-b0-finetuned-ade-512-512` with standard normalization; the phone's exact pipeline may differ.

**Run 3 plan (Indian frames, teacher labels only):**
- Frames: every 3rd unique frame per video. 100 whole videos held out for testing (never trained on):
  train 11,450 frames / 1,057 videos, test 1,056 frames / 100 videos.
- Removed before anything else: frames tagged with indoor-only classes (Elevator, Escalator, Couch, Chair,
  Pedestal fan: 82) and ATM (185, mostly inside booths).
- Confidence filter (frames and pixels): numbers chosen from two tests — A: teacher accuracy by confidence on
  Mapillary val (human labels); B: Indian frames by confidence band, judged by eye.
- After removal: train 11,211 frames, test 1,028 frames.
- **Test A** (kd/teacher_calibration.py, 300 Mapillary val photos): teacher right 98.9% when ≥ 95% sure (89.7% of
  pixels); 83% at 90–95%; 73% at 80–90%; ≤ 65% below 0.8. Keeping pixels ≥ 0.9 → 92% kept, 98.5% right;
  ≥ 0.8 → 94.5% kept, 97.9% right. All Mapillary photos average ≥ 0.80 confidence (weakest 0.80–0.85: 72% right).
  The teacher is much less sure on Indian frames (24-frame test: average 0.88), so the filter matters more there.
- **Test B** (kd/confidence_review.py, 2,005 Indian train frames): photos by average confidence — ≥ 0.90: 64%,
  0.85–0.90: 17%, 0.80–0.85: 9%, < 0.80: 10%. About 27% of Indian pixels are < 0.9 sure (Mapillary: 8%).
  The unsure pixels are mostly **ground** (mud, dirt, concrete yards, road edges): the teacher can't decide
  road / sidewalk / terrain there.
- **Walkable matters:** in India only sidewalk / pavement is walkable; road has vehicles and terrain is rough
  ground. Mixing these classes is dangerous, so unsure ground must not be learned.
  → **Filter: photos ≥ 0.80, pixels ≥ 0.9** (keeps ~70% of the Indian data).
- Root cause: Indian frames don't fit Mapillary (ground types, missing classes such as open drain / speed breaker
  / mud path, head camera looking down, scenes vary a lot by place). Distillation only passes on what the teacher
  knows. Human Indian labels (Haidata) are for the full version, not Lite → **Option A: run 3 now** for what the
  teacher is sure about (Indian vehicles, buildings, clear roads and lanes); walkable ground stays a known gap.
- App note (not changed here): `hsv2-lite-app/engine/core/nav_classes.h` `navToSemRole` marks floor, road,
  sidewalk **and terrain** as walkable (on main and origin/main; from the hslite scaffold). Wrong for India;
  needs an AUDIT row / owner decision.
- Start from a COPY of the KD student (`saved_models/mapillary_kd_a0.5`); compare run 3 against it and keep the
  better one.
- **Run 3 settings:** learning rate 1.2e-5 (5× smaller than runs 1–2, to keep what the model knows), 20,000 steps,
  batch 8 = 4 Mapillary (CE + KD as before) + 4 Indian (KD only, cells where teacher ≥ 0.9; label all ignore),
  alpha 0.5, T 2, same 4:3 cut / stretch / zoom / crop / flip for both (kd/dataset.py `_Augment`, `IndianSet`).
- Tests: line-up on Indian frames — photo, teacher map and learned area line up; padding never learned.
  20-step trial from the saved KD student: runs, loss ~0.57 (CE 0.20, KD 0.95), GPU 3.1 GB;
  **saved model checksum unchanged** before/after.

**Run 3 result** (10,144 Indian frames passed the photo filter; 20,000 steps, 1 h 9 min + validation; saved model
checksum unchanged):

| | KD student (saved) | Run 3 |
|---|---|---|
| Mapillary val mIoU (human labels) | 0.5354 | **0.5431** (best at step 12,000) |
| Held-back Indian videos (1,028 frames): agreement with teacher where teacher ≥ 0.9 sure (75.7% of cells) | 89.7% | **95.3%** |
| … mean IoU vs teacher | 0.479 | **0.609** |

- Mapillary did not drop; it rose slightly. **Animal 0.00 → 0.25 on Mapillary human labels** (Indian frames have
  many cows / dogs); on Indian test animal 0 → 0.70.
- Indian test gains: road 0.80 → 0.93, sidewalk 0.61 → 0.73, terrain 0.73 → 0.95, person 0.56 → 0.79,
  bus 0.56 → 0.80, other vehicle 0.49 → 0.70, trash can 0.26 → 0.54, traffic light 0.07 → 0.39.
- Small losses: Mapillary manhole 0.363 → 0.331, barrier 0.549 → 0.531, motorcycle 0.495 → 0.474;
  Indian bench 0.13 → 0.01 (rare). Pothole and curb cut still 0.
- Caveats: the Indian numbers compare with the teacher (run 3 was trained to copy it), not with the truth; part of
  the Mapillary gain may come from the extra training at a small learning rate, not the Indian frames.
- Saved as `saved_models/run3_indian/` (best.pt from step 12,000, checksum matches the run folder).
- **Pictures** (kd/compare_frames.py, app's 15 navigation classes, agreement with teacher):

| Frames | KD student (old) | Run 3 | B1 Cityscapes | B0 ADE (phone) |
|---|---|---|---|---|
| 24 from the held-back videos (fair) | 82.0% (walkable 84.7%) | **89.8%** (walkable 90.1%) | 71.2% | 73.6% |
| The earlier 24 (mostly from training videos) | 78.0% | 83.9% | 63.8% | 67.9% |

  Seen: dirt field called "road" by the old model → "terrain" in run 3; goat called "person" by the old model →
  animal in run 3. Worse: a park dirt footpath next to a railing — run 3 calls part of it "road" (teacher and old
  model: sidewalk). Walkable ground on Indian frames is still not reliable. Batches 50% Mapillary (human labels +
  teacher) / 50% Indian (teacher only). One run. Discard if Mapillary val drops.

## Status: distillation done (2026-11-22)
- **Final model: run 3** (`saved_models/run3_indian/best.pt`): Mapillary val mIoU 0.543; best of all runs on the
  held-back Indian videos and in the pictures.
- Stop here: a smaller student (B0) would lose accuracy, and further gains need real labeled Indian data —
  the Mapillary teacher has given what it can (unsure on Indian ground, no Indian-only classes).
- Known gaps: walkable ground on Indian frames (sidewalk vs road vs terrain) not reliable; pothole and curb cut 0;
  the app's walkable rule (road + terrain) is wrong for India.
- **Clean-up (2026-11-22):** training data deleted — Mapillary dataset, Mapillary + Indian teacher caches, Indian
  frames, test pictures, calibration raws, phone test data. All models kept (saved models, run folders, test
  runs, ONNX, phone builds). Retraining needs the dataset re-copied (~4 h) and the caches rebuilt (~3 h);
  Indian frames are still in S3.

## Final comparison (same 200 phone-like Mapillary val photos, 512×512, scored vs human labels)
| Model | Where | mIoU | Speed (NPU, this phone) | Size |
|---|---|---|---|---|
| Mask2Former Swin-L (teacher) | server GPU (phone: fails) | **0.5945** | — | 800 MB fp16 |
| **Run 3 w8a16** | phone | **0.5238** | 78.5 ms | 16.8 MB |
| Run 3 fp16 | phone | 0.5254 | 113.3 ms | 30.5 MB |
| KD student w8a16 | phone | 0.4970 | 77.9 ms | 16.8 MB |
| KD student fp16 | phone | 0.4986 | 110 ms | 30.5 MB |
| B0 ADE (phone today) | phone | — (other classes) | 47 / 65 ms (w8a16 / fp16) | 5.2 / 8.6 MB |

- Run 3 on the phone ≈ 88% of the teacher's mIoU at 1/50 of its size; NPU conversion costs ≤ 0.002 mIoU.
- **Mask2Former cannot run on this phone:** the fp16 context binary (built separately, 512×512, 100 query outputs)
  loads, then the NPU process crashes during execution (`err 1003`, CDSP restart). The log estimates ~1.9 GB of NPU
  process memory (weights 665 MB, spill-fill 841 MB, runlist 472 MB) — too much for the unsigned PD on a retail
  phone. The phone recovers by itself.

**Video test** (kd/phone_video.py): Corridor.mp4 (640×480, 30 fps, 2,632 frames, campus paved walkway) through
run 3 w8a16 on the phone NPU, app's preprocessing (nearest 512×512): **78.4 ms/frame average, stable over all
frames.** Buildings, trees, sky, people and raised curbs look right. **The wide paved walkway is mostly called
"road"** (only edges "sidewalk") — the Mapillary bias (wide paved surface = road); same walkable gap as above.

## Export to the phone NPU (KD student)
- Model: KD student (`saved_models/mapillary_kd_a0.5/best.pt`). Not public: binaries stay on the conversion server
  with a private copy on the PC, never in a repo.
- ONNX (kd/export_onnx.py, opset 17, onnxsim): same layout as the phone's SegFormer-B0 — input `image`
  1×512×512×3 RGB 0–1 (colour normalization inside), output `class_logits` 1×128×128×31, channel-last.
  ONNX vs PyTorch: 100% class-map match on 8 Indian frames (max logit diff 4e-5).
- Calibration (kd/build_calib.py): 200 phone-like images = 176 Mapillary train + 24 Indian frames.
  Accuracy checks use Mapillary val photos, never used for calibration.
- Conversion (kd/npu_convert.sh): QAIRT 2.47, existing export venv used read-only, `--preserve_io layout`,
  same HTP settings as B0 (v81, soc_model 87, O3, VTCM 8 MB; fp16 relaxed precision for fp16).

| Build | Size | B0 (phone today) |
|---|---|---|
| fp16 | 30.5 MB | 8.6 MB |
| w8a16 (8-bit weights, 16-bit activations, per-channel, CLE) | 16.8 MB | 5.2 MB |

**Phone test** (kd/phone_test.py; OnePlus 15 CPH2745, SM8850, Android 16, USB; qnn-net-run, burst, 100 runs;
NPU time from the profile log). Clean run: all HearSight apps uninstalled, no battery saver, thermal status 0.
(Runs with the app installed gave the same B0 numbers, so the app was not the cause.)

| Model | NPU avg | min | max |
|---|---|---|---|
| B0 fp16 (phone today) | 64.5 ms | 62.6 | 67.6 |
| B0 w8a16 | 47.0 ms | 44.1 | 74.3 |
| **B1 fp16 (ours)** | 110.0 ms | 93.2 | 124.2 |
| **B1 w8a16 (ours)** | 78.2 ms | 59.8 | 89.4 |

| Accuracy, 200 Mapillary val photos (phone-like, not used for calibration) | Class map = ONNX | mIoU phone | mIoU ONNX |
|---|---|---|---|
| B1 fp16 | 99.97% | 0.4986 | 0.4986 |
| B1 w8a16 | 99.34% (worst photo 93.3%) | 0.4970 | 0.4986 |

- Quantization costs almost nothing: w8a16 loses 0.002 mIoU. (0.499 on these 200 photos vs 0.537 on all 2,000:
  subset difference, same model.)
- Speed: B1 ≈ 1.7× B0 (fp16 110 vs 65 ms; w8a16 78 vs 47 ms). Same 1.5–1.7× in every run.
- The earlier cloud figures for B0 (12.6 / 10.0 ms, Galaxy S26, runtime 2.50) are ~5× lower than this real
  OnePlus 15 with qnn-net-run 2.47.
- **Why (option C):** on this retail phone the NPU runs in *unsigned PD* mode (log: `Unsigned:Y, Signed:N`) and
  the clock vote is refused (`Access denied … persist.vendor.fastrpc.process.clockvote`). Every perf profile gives
  the same B0 time (low_power_saver 64.7, balanced 64.5, burst 64.4 ms) → the NPU clock is fixed at its default.
  Cloud test phones are not limited this way. A normal app is also unsigned, so it very likely gets the same speed —
  not yet confirmed inside the HearSight app (its engine logs NPU time).
- Spec goal: segmentation ≥ 10 Hz (≤ 100 ms/frame). B1 w8a16 (78 ms) meets it alone; B1 fp16 (110 ms) does not.
