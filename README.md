<<<<<<< HEAD
Hybrid CNN-Transformer Segmentation

This small package provides a hybrid U-Net that uses a Transformer bottleneck (Vision-Transformer-style) for improved global context, suitable for brain MRI tumor segmentation.

Files

- `segmentation_model.py` : builds the hybrid U-Net + Transformer bottleneck
- `losses_metrics.py` : Dice metric and BCE+Dice loss
- `train.py` : dataset pipeline and training script
- `requirements.txt` : Python dependencies

Quick start

1. Install dependencies (preferably in a venv):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepare dataset directories:

- Put RGB images (256x256 or any size) in `C:\path\to\images`.
- Put corresponding masks in `C:\path\to\masks` (masks should be single-channel; filenames must share a base name with the image or contain the image base name).

3. Run training:

```powershell
python train.py --images_dir "C:\path\to\images" --masks_dir "C:\path\to\masks" --batch_size 8 --epochs 30
```

Notes

- The training script resizes inputs to 256×256 and thresholds masks to binary.
- The code assumes masks file names contain the image base name (examples: `case_0001.tif` and `case_0001_mask.tif`).
- The Transformer bottleneck expects the bottleneck spatial dims to be divisible by the patch size. The default patch size used in `segmentation_model.py` is 2.

Next steps / improvements

- Add augmentations (albumentations) inside the tf.data pipeline.
- Add mixed-precision training and dataset caching for speed.
- Tweak the Transformer depth/heads/MLP sizes for your dataset and GPU memory.

If you want, I can also:

- Add an evaluation script that computes Dice over the test set and saves some overlay visualizations.
- Convert the model to use a pretrained MobileViT / ViT encoder from TensorFlow Hub.
=======
# machine-learning-project
This respository describes the model that i built as part of my machine learning course in masters
>>>>>>> 833388285f19062fc5cdb1e46d482abefff5d2ec
