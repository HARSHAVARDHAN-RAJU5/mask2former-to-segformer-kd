# mask2former-to-segformer-kd

Distil Mask2Former Swin-L (Mapillary Vistas v1.2) into a small SegFormer for outdoor segmentation on the phone (HSV2 Lite).

## Tools
- `tools/make_slim_zip.py`: make a smaller Vistas zip with only train/val images and v1.2 labels.
- `tools/upload_resume.py`: upload a big file to the server over SSH, with a progress bar, resume after network cuts and a checksum check.
