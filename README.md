# Tiny-MAE-ViT

Masked Autoencoder with Vision Transformers trained on Tiny-ImageNet

## Overview

This project implements a **Masked Autoencoder (MAE)** using **Vision Transformers (ViT)** for self-supervised learning on image data.

The model learns to reconstruct missing patches of an image after randomly masking a large portion of the input. This allows the model to learn **strong visual representations without labeled data**.

The implementation follows the MAE concept introduced by Meta AI in the paper:

> **Masked Autoencoders Are Scalable Vision Learners**
> Kaiming He et al.

The model is trained on the **Tiny-ImageNet dataset** and demonstrates how transformer-based encoders can learn meaningful visual features through reconstruction tasks.

---

# Features

* Vision Transformer encoder
* Lightweight transformer decoder
* Random patch masking
* Patch embedding system
* Training on Tiny-ImageNet
* Reconstruction visualization
* Interactive demo using Gradio

---

# Architecture

The system follows the standard **Masked Autoencoder pipeline**:

1. Image is split into patches
2. A large portion of patches are randomly masked
3. Visible patches are passed through the **ViT Encoder**
4. A **Transformer Decoder** reconstructs the missing patches
5. Reconstruction loss is computed only on masked patches

Pipeline:

Image → Patch Embedding → Random Masking → ViT Encoder → Decoder → Reconstructed Image

---

# Project Structure

```
tiny-mae-vit/
│
├── mae-assignment.ipynb        # Full implementation notebook
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── outputs/                    # Reconstruction results
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/tiny-mae-vit.git
cd tiny-mae-vit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision timm scikit-image gradio kagglehub
```

---

# Dataset

This project uses **Tiny-ImageNet**.

The notebook automatically downloads it using:

```python
kagglehub.dataset_download("akash2sharma/tiny-imagenet")
```

Dataset structure:

```
tiny-imagenet
 ├── train
 ├── val
 └── test
```

---

# How the Model Works

## 1. Image Patching

Images are divided into fixed-size patches (similar to tokens in NLP).

Example:

```
224x224 image
↓
16x16 patches
↓
196 patch tokens
```

Each patch is flattened and converted into an embedding.

---

## 2. Random Masking

A large percentage of patches (typically **75%**) are randomly removed.

Example:

```
Original patches: 196
Visible patches: 49
Masked patches: 147
```

Only visible patches are sent to the encoder.

---

## 3. ViT Encoder

The encoder is a **Vision Transformer (ViT-Base)** that processes only the visible patches.

Benefits:

* Lower compute cost
* Forces model to learn strong representations

---

## 4. Transformer Decoder

The decoder receives:

* encoded visible patches
* mask tokens for missing patches

It attempts to reconstruct the original image.

---

## 5. Loss Function

Loss is calculated **only on masked patches**:

```
Loss = MSE(reconstructed_pixels, original_pixels)
```

This encourages the network to predict missing visual information.

---

# Training the Model

Run the notebook:

```
mae-assignment.ipynb
```

Training steps include:

1. Dataset loading
2. Patch creation
3. Masking
4. Encoder forward pass
5. Decoder reconstruction
6. Loss calculation
7. Optimization

---

# Visualizing Reconstruction

The model can reconstruct masked images.

Example pipeline:

```
Input Image
↓
Random Mask Applied
↓
Model Reconstruction
↓
Reconstructed Image
```

This helps evaluate how well the model learned visual features.

---

# Interactive Demo

The notebook includes a **Gradio interface** that allows users to:

* Upload an image
* Apply masking
* View reconstruction

Run the demo:

```python
import gradio as gr
```

---

# How to Build Your Own MAE Model

Follow these steps:

### Step 1 — Prepare Dataset

Choose a dataset such as:

* Tiny ImageNet
* CIFAR-10
* ImageNet
* Custom dataset

Load images using PyTorch DataLoader.

---

### Step 2 — Create Image Patches

Split images into patches:

```
Image → patches → flattened vectors
```

Use a **patch embedding layer**.

---

### Step 3 — Apply Random Masking

Randomly remove a large portion of patches.

Typical masking ratio:

```
mask_ratio = 0.75
```

---

### Step 4 — Build the Encoder

Use a Vision Transformer:

```
ViT Base
12 layers
768 hidden dimension
12 attention heads
```

The encoder processes **only visible patches**.

---

### Step 5 — Add Decoder

Create a lightweight transformer decoder.

Purpose:

* reconstruct masked patches

Decoder is smaller than encoder.

---

### Step 6 — Reconstruction Loss

Use **Mean Squared Error (MSE)** between:

```
predicted_pixels
vs
original_pixels
```

Loss is computed **only on masked patches**.

---

### Step 7 — Train Model

Train with:

```
optimizer = AdamW
learning_rate = 1e-4
batch_size = 64
epochs = 100
```

---

### Step 8 — Visualize Results

Reconstruct masked images and compare with original.

This verifies the model learned useful features.

---

# Applications

Masked Autoencoders are used for:

* Self-supervised learning
* Representation learning
* Image pretraining
* Transfer learning
* Vision transformers pretraining

---

# Future Improvements

Possible extensions:

* Train on full ImageNet
* Fine-tune for classification
* Use larger ViT models
* Improve decoder architecture
* Add distributed training

---

# References

MAE Paper:
https://arxiv.org/abs/2111.06377

Vision Transformer:
https://arxiv.org/abs/2010.11929

---

# Author

Your Name

GitHub:
https://github.com/yourusername
