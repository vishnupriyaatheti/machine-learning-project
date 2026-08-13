{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1c106811",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "import albumentations as A\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34827515",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Tumour slices     : 1373\n",
      "Non-tumour slices : 2556\n"
     ]
    }
   ],
   "source": [
    "\n",
    "base_dir = r\"C:/Users/athet/Downloads/archive/kaggle_3m\"\n",
    "\n",
    "tumour = []\n",
    "nontumour = []\n",
    "\n",
    "for patient in os.listdir(base_dir):\n",
    "    patient_path = os.path.join(base_dir, patient)\n",
    "    if not os.path.isdir(patient_path):\n",
    "        continue\n",
    "\n",
    "    for file in os.listdir(patient_path):\n",
    "        if file.endswith(\".tif\") and not file.endswith(\"_mask.tif\"):\n",
    "            img_path = os.path.join(patient_path, file)\n",
    "            mask_path = img_path.replace(\".tif\", \"_mask.tif\")\n",
    "\n",
    "            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)\n",
    "\n",
    "            if np.sum(mask) > 0:\n",
    "                tumour.append((img_path, mask_path))\n",
    "            else:\n",
    "                nontumour.append((img_path, mask_path))\n",
    "\n",
    "print(\"Tumour slices     :\", len(tumour))\n",
    "print(\"Non-tumour slices :\", len(nontumour))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "28696d3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "IMG_SIZE = 256\n",
    "\n",
    "def preprocess_image(img):\n",
    "    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))\n",
    "    return img.astype(\"float32\") / 255.0\n",
    "\n",
    "def preprocess_mask(mask):\n",
    "    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))\n",
    "    return (mask > 0).astype(\"float32\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "f8fcb3fe",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\athet\\mycnnenv\\Lib\\site-packages\\albumentations\\core\\validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.\n",
      "  original_init(self, **validated_kwargs)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "augmentor = A.Compose([\n",
    "    A.HorizontalFlip(p=0.5),\n",
    "    A.VerticalFlip(p=0.5),\n",
    "    A.RandomRotate90(p=0.5),\n",
    "    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.7),\n",
    "    A.ElasticTransform(p=0.3),\n",
    "    A.RandomBrightnessContrast(p=0.3),\n",
    "])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a3489fdd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Augmenting tumour slices: 1183\n"
     ]
    }
   ],
   "source": [
    "balanced_images = []\n",
    "balanced_masks = []\n",
    "\n",
    "# Add all non-tumour first\n",
    "for img_path, mask_path in nontumour:\n",
    "    img = cv2.imread(img_path)\n",
    "    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)\n",
    "\n",
    "    balanced_images.append(preprocess_image(img))\n",
    "    balanced_masks.append(preprocess_mask(mask))\n",
    "\n",
    "# Add original tumour slices\n",
    "for img_path, mask_path in tumour:\n",
    "    img = cv2.imread(img_path)\n",
    "    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)\n",
    "\n",
    "    balanced_images.append(preprocess_image(img))\n",
    "    balanced_masks.append(preprocess_mask(mask))\n",
    "\n",
    "# Now augment tumour until counts match\n",
    "\n",
    "\n",
    "# Determine target = bigger class count\n",
    "target = max(len(tumour), len(nontumour))\n",
    "\n",
    "# Only augment tumour (minority class)\n",
    "needed = target - len(tumour)\n",
    "\n",
    "print(\"Augmenting tumour slices:\", needed)\n",
    "\n",
    "\n",
    "for _ in range(needed):\n",
    "    img_path, mask_path = random.choice(tumour)\n",
    "\n",
    "    img = cv2.imread(img_path)\n",
    "    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)\n",
    "\n",
    "    transformed = augmentor(image=img, mask=mask)\n",
    "\n",
    "    balanced_images.append(preprocess_image(transformed[\"image\"]))\n",
    "    balanced_masks.append(preprocess_mask(transformed[\"mask\"]))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "f6d27aa7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Balanced dataset created.\n",
      "len of balanced images: 5112\n",
      "len of balanced masks : 5112\n",
      "Final image shape: (5112, 256, 256, 3)\n",
      "Final mask shape : (5112, 256, 256)\n"
     ]
    }
   ],
   "source": [
    "balanced_images = np.array(balanced_images)\n",
    "balanced_masks  = np.array(balanced_masks)\n",
    "\n",
    "print(\"Balanced dataset created.\")\n",
    "print(\"len of balanced images:\", len(balanced_images))\n",
    "print(\"len of balanced masks :\", len(balanced_masks))\n",
    "print(\"Final image shape:\", balanced_images.shape)\n",
    "print(\"Final mask shape :\", balanced_masks.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "28bddb59",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.savez_compressed(\"brainMRI_balanced.npz\",\n",
    "                    images=balanced_images,\n",
    "                    masks=balanced_masks)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "mycnnenv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
