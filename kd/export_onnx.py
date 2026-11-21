"""Export a saved student to ONNX with the same input/output layout as the phone's SegFormer-B0.

  input  "image"         1 x 512 x 512 x 3   RGB in [0, 1], channel-last (normalization is inside the model)
  output "class_logits"  1 x 128 x 128 x 31  channel-last; argmax over the last axis = our class id

Then checks ONNX (onnxruntime) against PyTorch on a few phone-like frames.

Usage (on the server):
  python3 kd/export_onnx.py --ckpt <best.pt> --out <model.onnx> --check-frames <folder of images>
"""

import argparse
import os
import sys

os.environ.setdefault("HF_HOME", "/mnt/nunnari3/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kd import classes as C  # noqa: E402
from kd.dataset import CAMERA, MEAN, STD, box, cut_4_3  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import SegformerForSemanticSegmentation  # noqa: E402

SIDE = 512


class PhoneSegformer(torch.nn.Module):
    """RGB 0..1 channel-last in -> logits channel-last out; normalization inside."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", MEAN.view(1, 3, 1, 1).clone())
        self.register_buffer("std", STD.view(1, 3, 1, 1).clone())

    def forward(self, image):
        x = (image.permute(0, 3, 1, 2) - self.mean) / self.std
        return self.model(pixel_values=x).logits.permute(0, 2, 3, 1)


def load_student(ckpt):
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-cityscapes-1024-1024", num_labels=C.NUM_CLASSES,
        id2label=dict(enumerate(C.NAMES)), label2id={n: i for i, n in enumerate(C.NAMES)},
        ignore_mismatched_sizes=True, attn_implementation="eager")
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return m.eval()


def phone_frame(path):
    """Phone-like input as float32 1x512x512x3 in [0,1]: centred 4:3 cut -> 640x480 -> nearest 512x512."""
    img = Image.open(path).convert("RGB")
    u, f = cut_4_3(*img.size, random_pos=False)
    cam = img.resize(CAMERA, Image.BILINEAR, box=box(u, f, img.size))
    return np.asarray(cam.resize((SIDE, SIDE), Image.NEAREST), dtype=np.float32)[None] / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-frames", required=True)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    net = PhoneSegformer(load_student(a.ckpt)).eval()
    dummy = torch.rand(1, SIDE, SIDE, 3)
    torch.onnx.export(net, (dummy,), a.out, input_names=["image"], output_names=["class_logits"],
                      opset_version=a.opset, do_constant_folding=True, dynamo=False)

    import onnx
    from onnxsim import simplify
    model, ok = simplify(onnx.load(a.out))
    assert ok, "onnxsim check failed"
    onnx.save(model, a.out)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(model.graph.input) + list(model.graph.output)}
    print(f"Saved {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB), opset {a.opset}, {len(model.graph.node)} nodes, I/O {shapes}")

    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    names = sorted(os.listdir(a.check_frames))[:8]
    diffs, match = [], []
    for n in names:
        x = phone_frame(os.path.join(a.check_frames, n))
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"image": x})[0]
        diffs.append(np.abs(ref - got).max())
        match.append((ref.argmax(-1) == got.argmax(-1)).mean())
    print(f"ONNX vs PyTorch on {len(names)} frames: max logit diff {max(diffs):.2e}, "
          f"class map match {np.mean(match) * 100:.3f}% (worst {min(match) * 100:.3f}%)")


if __name__ == "__main__":
    main()
