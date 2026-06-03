# CNN Autoencoder for EMNIST

This project explores the use of Autoencoders, specifically Convolutional Neural Networks (CNNs), to reconstruct images from the EMNIST dataset. We compare simple and deeper CNN architectures to evaluate their performance in compressing and reconstructing handwritten characters.

## Dataset

We use the EMNIST (Extended MNIST) Balanced dataset, which includes both letters and digits:
-   **Training Set**: 112,800 images
-   **Test Set**: 18,800 images
-   **Image Size**: 28x28 pixels (grayscale)

## Models

We trained two CNN-based autoencoder models to compare their reconstruction capabilities:

### Model 1: Simple CNN Autoencoder
A straightforward architecture to establish a baseline.

**Encoder:**
-   `Conv2d(1, 16, kernel_size=3, padding=1)`
-   `ReLU`
-   `MaxPool2d(2, 2)`
-   `Conv2d(16, 32, kernel_size=3, padding=1)`
-   `ReLU`
-   `MaxPool2d(2, 2)`
-   `Conv2d(32, 64, kernel_size=3, padding=1)` (Bottleneck)
-   `ReLU`

**Decoder:**
-   `ConvTranspose2d(64, 32, kernel_size=2, stride=2)`
-   `ReLU`
-   `ConvTranspose2d(32, 16, kernel_size=2, stride=2)`
-   `ReLU`
-   `Conv2d(16, 1, kernel_size=3, padding=1)`
-   `Sigmoid`

**Total Parameters:** 33,729

### Model 2: Deeper CNN Autoencoder with Batch Normalization
A more complex model with additional layers and batch normalization to improve training stability and reconstruction quality.

**Encoder:**
-   `Conv2d(1, 32, kernel_size=3, padding=1)`
-   `BatchNorm2d(32)`
-   `ReLU`
-   `MaxPool2d(2, 2)`
-   `Conv2d(32, 64, kernel_size=3, padding=1)`
-   `BatchNorm2d(64)`
-   `ReLU`
-   `MaxPool2d(2, 2)`
-   `Conv2d(64, 128, kernel_size=3, padding=1)` (Bottleneck)
-   `BatchNorm2d(128)`
-   `ReLU`

**Decoder:**
-   `ConvTranspose2d(128, 64, kernel_size=2, stride=2)`
-   `BatchNorm2d(64)`
-   `ReLU`
-   `ConvTranspose2d(64, 32, kernel_size=2, stride=2)`
-   `BatchNorm2d(32)`
-   `ReLU`
-   `Conv2d(32, 1, kernel_size=3, padding=1)`
-   `Sigmoid`

**Total Parameters:** 134,657

## Results

Both models were trained using the Adam optimizer and Mean Squared Error (MSE) loss. The deeper architecture with batch normalization demonstrated significantly better reconstruction quality.

| Model | Epochs | Final Training Loss | Final Validation Loss |
| :--- | :---: | :---: | :---: |
| **Model 1 (Simple CNN)** | 10 | 0.001033 | 0.000988 |
| **Model 2 (Deeper + BatchNorm)** | 8 | 0.000313 | 0.000349 |

The deeper model (Model 2) is the superior model based on the lower validation loss.

## Usage

The project is structured within Jupyter Notebooks. To run the code:

1.  Ensure you have the required libraries installed (`numpy`, `pandas`, `matplotlib`, `torch`, `sklearn`).
2.  Download the `emnist-balanced-train.csv`, `emnist-balanced-test.csv`, and `emnist-balanced-mapping.txt` files and place them in the project directory.
3.  Execute the cells in the notebook `cnn.ipynb` to train the models and visualize the results.
