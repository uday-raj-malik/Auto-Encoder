# Autoencoder Comparison on EMNIST-Balanced
A comparative study of three autoencoder architectures — two fully connected (ANN-based) and two convolutional (CNN-based) — trained on the EMNIST Balanced dataset for unsupervised character representation learning.
 
## 📁 Project Structure
```
├── ann_1.ipynb       # Shallow ANN Autoencoder (bottleneck: 64)
├── ann_2.ipynb       # Deep ANN Autoencoder (bottleneck: 32)
└── cnn.ipynb         # CNN Autoencoders — AE_CNN_1 and AE_CNN_2
```
 
## 📊 Dataset
**EMNIST Balanced** — an extension of MNIST containing handwritten digits and uppercase/lowercase letters.
- Input: 28×28 grayscale images (flattened to 784 for ANN, kept as 2D for CNN)
- Classes: 47 balanced classes
- Preprocessing: pixel normalization to [0, 1], rotation fix (90° clockwise + horizontal flip)
Download the dataset from: [EMNIST Official Page](https://www.nist.gov/itl/products-and-services/emnist-dataset)
 
Place these files in the project root before running:
```
emnist-balanced-train.csv
emnist-balanced-test.csv
emnist-balanced-mapping.txt
```
 
## 🧠 Model Architectures
 
### ANN Autoencoder 1 (`ann_1.ipynb`)
Shallow fully-connected autoencoder.
 
| Layer | Size |
|------------|----------------------------------|
| Encoder | 784 → 256 → 128 → **64** |
| Decoder | 64 → 128 → 256 → 784 |
| Activation | ReLU (encoder), Sigmoid (output) |
| Batch Size | 64 |
| Bottleneck | 64 |
 
### ANN Autoencoder 2 (`ann_2.ipynb`)
Deeper fully-connected autoencoder with a tighter bottleneck.
 
| Layer | Size |
|------------|--------------------------------------|
| Encoder | 784 → 512 → 256 → 128 → 64 → **32** |
| Decoder | 32 → 64 → 128 → 256 → 512 → 784 |
| Activation | ReLU (encoder), Sigmoid (output) |
| Batch Size | 32 |
| Bottleneck | 32 |
 
### CNN Autoencoder 1 — `AE_CNN_1` (`cnn.ipynb`)
Simple convolutional autoencoder without batch normalization.
 
| Stage | Operation | Output Shape |
|---------|---------------------------|---------------|
| Encoder | Conv(1→16) + MaxPool | 16 × 14 × 14 |
| | Conv(16→32) + MaxPool | 32 × 7 × 7 |
| | Conv(32→64) ← bottleneck | 64 × 7 × 7 |
| Decoder | ConvTranspose(64→32) | 32 × 14 × 14 |
| | ConvTranspose(32→16) | 16 × 28 × 28 |
| | Conv(16→1) + Sigmoid | 1 × 28 × 28 |
 
### CNN Autoencoder 2 — `AE_CNN_2` (`cnn.ipynb`)
Deeper CNN autoencoder with BatchNorm for stable training.
 
| Stage | Operation | Output Shape |
|---------|--------------------------------|---------------|
| Encoder | Conv(1→32) + BN + MaxPool | 32 × 14 × 14 |
| | Conv(32→64) + BN + MaxPool | 64 × 7 × 7 |
| | Conv(64→128) + BN ← bottleneck | 128 × 7 × 7 |
| Decoder | ConvTranspose(128→64) + BN | 64 × 14 × 14 |
| | ConvTranspose(64→32) + BN | 32 × 28 × 28 |
| | Conv(32→1) + Sigmoid | 1 × 28 × 28 |
 
## ⚙️ Training Setup
 
| Hyperparameter | ANN-1 | ANN-2 | CNN-1 & CNN-2 |
|----------------|---------|---------|----------------|
| Loss | MSELoss | MSELoss | MSELoss |
| Optimizer | Adam | Adam | Adam |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| Epochs | 5 | 5 | 10 |
| Batch Size | 64 | 32 | 64 |
 
## 📈 Evaluation & Visualizations
Each notebook includes:
- **Reconstruction quality** — original vs reconstructed image grids
- **Validation loss curve** across epochs
- **PCA plot** of the latent space (full test set)
- **t-SNE plot** of the latent space (5,000-sample subset, perplexity=30)
## 🛠️ Requirements
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```
 
## 🚀 Usage
Open any notebook in Jupyter and run all cells in order:
```bash
jupyter notebook ann_1.ipynb
jupyter notebook ann_2.ipynb
jupyter notebook cnn.ipynb
```
Ensure the EMNIST CSV files are in the same directory as the notebooks.
 
## 📌 Key Observations
- ANN autoencoders flatten spatial structure; CNN autoencoders preserve it via convolutional layers.
- The deeper ANN (ann_2) uses a tighter bottleneck (32-dim) vs ann_1 (64-dim).
- AE_CNN_2 adds BatchNorm to each layer for more stable training over 10 epochs.
- Latent space visualizations (PCA / t-SNE) reveal how well each architecture clusters character classes.
