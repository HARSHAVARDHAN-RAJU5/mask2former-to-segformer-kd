# Mask2Former → SegFormer: outdoor segmentation for the HearSight phone

Knowledge distillation of a large **Mask2Former Swin-L** teacher (trained on Mapillary Vistas v1.2) into a small
**SegFormer-B1** student that runs on the **OnePlus 15 NPU** (Snapdragon SM8850, Hexagon v81) in the
HSV2 Lite navigation aid for blind and low-vision users.

**Final model:** SegFormer-B1, 31 outdoor classes, 512×512 input — **78 ms per frame** on the phone NPU (8-bit),
about **88% of the teacher's accuracy at 1/50 of its size**.

Every test and decision is recorded in [`decision.md`](decision.md).

---

## Results

Mapillary Vistas validation, phone-like 512×512 input, scored against human labels (31 classes):

| Model | Runs on | mIoU (200 photos) | mIoU (2,000 photos) | Pixel acc. | Size |
|---|---|---|---|---|---|
| Mask2Former Swin-L (teacher) | server GPU | **0.595** | — | 92.8% | 800 MB |
| **SegFormer-B1, run 3 (final)** | **phone NPU** | **0.525** | **0.543** | 92.6% | **17 MB** (8-bit) |
| SegFormer-B1, KD student | phone NPU | 0.499 | 0.535 | 92.6% | 17 MB |
| SegFormer-B1, no teacher (baseline) | phone NPU | — | 0.537 | 92.7% | 17 MB |

On the phone (OnePlus 15, NPU time per frame, 100 runs):

| Model | 8-bit (w8a16) | fp16 |
|---|---|---|
| **SegFormer-B1, run 3** | **78 ms** | 113 ms |
| SegFormer-B0 (current app model, ADE20K) | 47 ms | 65 ms |
| Mask2Former Swin-L | does not fit (~1.9 GB NPU memory) | — |

- Conversion to the NPU costs ≤ 0.002 mIoU.
- On 100 held-back Indian street videos, run 3 agrees with the teacher on 95% of confident pixels (KD student: 90%),
  and learned classes such as **animal** (0 → 0.70) that the Mapillary-only student missed.

---

## How it works

1. **Classes.** The teacher's 65 Mapillary classes are merged into **31** navigation-relevant classes
   (road, sidewalk, crosswalk, curb, curb cut, terrain, pothole, pole, person, vehicles …); teacher probabilities
   of merged classes are added up.
2. **Teacher cache.** The teacher runs once on every training photo (short side 1024); the top-5 classes per pixel
   are saved, so training never runs the big model again.
3. **Student training.** Loss = α · KD + (1 − α) · CE, with KD = T² · KL(teacher ‖ student) on soft labels
   (α = 0.5, T = 2). Ignored pixels are skipped in both.
4. **Phone-matched input.** The phone camera is 640×480 and the app stretches it to 512×512, so training does the
   same: wide photos are cut to 4:3, stretched to a square, randomly zoomed, cropped and flipped — identically for
   the photo, the label and the teacher map.
5. **Run 3: Indian streets.** Starting from the KD student, training continues on 50% Mapillary (human labels)
   + 50% unlabeled Indian street frames (teacher only). Only confident teacher answers are learned
   (frames ≥ 0.80 average confidence, pixels ≥ 0.9).
6. **Phone export.** ONNX with normalization inside (input RGB 0–1, NHWC 1×512×512×3; output 1×128×128×31) →
   QNN context binary (QAIRT 2.47, fp16 and w8a16) → tested on the phone with `qnn-net-run`.

---

## Repository

| Folder / file | What it does |
|---|---|
| `kd/classes.py` | The 31-class table and the 65 → 31 mapping (matched by class name) |
| `kd/teacher.py`, `teacher_check.py` | Load and check the teacher |
| `kd/teacher_cache.py` | Save the teacher's answers (resumable, logs failed photos) |
| `kd/teacher_calibration.py`, `confidence_review.py` | When the teacher's confidence can be trusted |
| `kd/dataset.py`, `kd/train.py` | Phone-matched data and the KD + CE training loop (resumable) |
| `kd/compare_models.py`, `compare_frames.py` | Compare models per class and in pictures |
| `kd/export_onnx.py`, `build_calib.py`, `npu_convert.sh`, `npu_validate.py` | Export and convert for the NPU |
| `kd/phone_test.py`, `phone_video.py` | Speed, accuracy and a full video on the phone |
| `kd/server_video.py` | The teacher on a video (server GPU) |
| `tools/` | Smaller dataset zip, resumable upload |
| `decision.md` | Every test, number and decision |

Model files and where they are stored are not in this repository.

---

## Known limitations

- **Walkable ground on Indian streets is not reliable:** wide paved walkways are often called "road", and
  sidewalk vs road vs terrain is uncertain. Fixing this needs human-labelled Indian data.
- **Pothole and curb cut are not learned** (too rare in Mapillary).
- Phone NPU times are ~5× slower than cloud-device benchmarks: on a retail phone the NPU runs in unsigned mode
  with a fixed clock.

## Licence

Mapillary Vistas is **CC BY-NC-SA** (non-commercial). Models trained on it need a licence check before use in a
product.
