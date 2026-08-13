import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from segmentation_model import build_hybrid_unet
from losses_metrics import bce_dice_loss, dice_coef


AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=str, required=True, help="Directory with input RGB images")
    p.add_argument("--masks_dir", type=str, required=True, help="Directory with corresponding masks (single-channel)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--save_dir", type=str, default="checkpoints")
    return p.parse_args()



def load_image_mask_pair(image_path, mask_path, img_size=(256, 256)):
    # Read image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0

    # Read mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask = tf.image.resize(mask, img_size)
    # Masks might have values 0/255 or 0/1; convert to binary 0/1
    mask = tf.cast(mask > 127, tf.float32)

    return image, mask


def make_dataset(image_paths, mask_paths, batch_size=8, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths))

    def _map_fn(img_p, m_p):
        img, m = load_image_mask_pair(img_p, m_p)
        return img, m

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def find_matching_mask(image_path, masks_dir):
    # Assumes masks use same base name + maybe suffix _mask
    base = os.path.splitext(os.path.basename(image_path))[0]
    candidates = os.listdir(masks_dir)
    for c in candidates:
        if base in c:
            return os.path.join(masks_dir, c)
    # fallback: try exact base + '_mask.tif'
    guess = base + '_mask.tif'
    gpath = os.path.join(masks_dir, guess)
    if os.path.exists(gpath):
        return gpath
    raise FileNotFoundError(f"No mask found for {image_path}")


def collect_pairs(images_dir, masks_dir):
    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
    mask_files = []
    for p in image_files:
        m = find_matching_mask(p, masks_dir)
        mask_files.append(m)
    return image_files, mask_files


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    image_paths, mask_paths = collect_pairs(args.images_dir, args.masks_dir)
    print(f"Found {len(image_paths)} image-mask pairs")

    train_imgs, test_imgs, train_masks, test_masks = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42
    )
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        train_imgs, train_masks, test_size=0.15, random_state=42
    )

    train_ds = make_dataset(train_imgs, train_masks, batch_size=args.batch_size, shuffle=True)
    val_ds = make_dataset(val_imgs, val_masks, batch_size=args.batch_size, shuffle=False)
    test_ds = make_dataset(test_imgs, test_masks, batch_size=args.batch_size, shuffle=False)

    model = build_hybrid_unet(input_shape=(256,256,3), num_classes=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=bce_dice_loss, metrics=[dice_coef])
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(args.save_dir, 'best_model.h5'), save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Evaluate
    print("Evaluating on test set:")
    res = model.evaluate(test_ds)
    print(res)


if __name__ == '__main__':
    main()
