# Autoencoder Models: ANN vs CNN

## Overview

This project implements and compares multiple Autoencoder architectures
using PyTorch for image reconstruction tasks. The objective is to learn
compressed latent representations of input images and reconstruct them
with minimum reconstruction error.

Three implementations are included:

-   ANN Autoencoder Model 1
-   ANN Autoencoder Model 2 (Deeper Architecture)
-   CNN Autoencoder Models

------------------------------------------------------------------------

## Tech Stack

-   Python
-   PyTorch
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn

------------------------------------------------------------------------

## Model Architectures

### 1. ANN Autoencoder 1

A fully connected neural network based autoencoder.

### Architecture

Encoder:

784 → 256 → 128 → 64

Decoder:

64 → 128 → 256 → 784

### Final Loss

  Metric            Loss
  ----------------- --------
  Training Loss     0.0086
  Validation Loss   0.0083

------------------------------------------------------------------------

## 2. ANN Autoencoder 2

A deeper ANN based autoencoder with a smaller bottleneck representation.

### Architecture

Encoder:

784 → 512 → 256 → 128 → 64 → 32

Decoder:

32 → 64 → 128 → 256 → 512 → 784

### Final Loss

  Metric            Loss
  ----------------- --------
  Training Loss     0.0203
  Validation Loss   0.0200

------------------------------------------------------------------------

## 3. CNN Autoencoder

Convolutional autoencoders were implemented to better preserve spatial
image features.

### CNN Model 1

Uses convolution, pooling and transpose convolution layers.

### Final Loss

  Metric            Loss
  ----------------- ----------
  Training Loss     0.001033
  Validation Loss   0.000988

------------------------------------------------------------------------

### CNN Model 2 (Deeper + Batch Normalization)

An improved CNN autoencoder with deeper feature extraction and Batch
Normalization.

### Final Loss

  Metric            Loss
  ----------------- ----------
  Training Loss     0.000313
  Validation Loss   0.000349

------------------------------------------------------------------------

## Model Comparison

  Model               Final Validation Loss
  ------------------- -----------------------
  ANN Autoencoder 1   0.0083
  ANN Autoencoder 2   0.0200
  CNN Autoencoder 1   0.000988
  CNN Autoencoder 2   0.000349

The CNN Autoencoder 2 achieved the best reconstruction performance with
the lowest validation loss.

------------------------------------------------------------------------

## Results

-   ANN models successfully learned compressed image representations.
-   CNN models performed significantly better due to their ability to
    capture spatial features.
-   The best performing model achieved a validation reconstruction loss
    of **0.000349**.

------------------------------------------------------------------------

## How to Run

1.  Install dependencies:

``` bash
pip install torch numpy pandas matplotlib scikit-learn
```

2.  Open the notebooks:

-   ann_1.ipynb
-   ann_2.ipynb
-   cnn.ipynb

3.  Run all cells to train and evaluate the models.

------------------------------------------------------------------------

## Future Improvements

-   Add denoising autoencoder functionality
-   Experiment with different latent dimensions
-   Add visualization of reconstructed outputs
-   Try Variational Autoencoders (VAE)
