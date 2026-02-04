# Eigenfaces Face Recognition

## Overview

The **Eigenfaces Face Recognition** system is a Python implementation of a classic computer vision algorithm, designed to identify human faces using statistical dimensionality reduction. This project explores the intersection of Linear Algebra and Machine Learning, demonstrating how high-dimensional image data can be compressed into a compact "face space" while preserving essential identifying features.

By implementing three distinct mathematical approaches to Principal Component Analysis (PCA), this system provides a comparative platform for analyzing algorithmic efficiency and numerical stability in biometric applications.

## Features

* **Multi-Method PCA:** Implements and compares three projection techniques: Standard Covariance, Dual (Snapshot) Method, and Singular Value Decomposition (SVD).
* **Face Recognition:** Identifies subjects by projecting test images into the latent subspace and minimizing Euclidean distance.
* **Image Reconstruction:** Visualizes the quality of compression by reconstructing images from their eigen-weights and calculating the Mean Squared Error (MSE).
* **Performance Profiling:** Includes custom decorators to measure execution time, highlighting the computational trade-offs between different PCA methods.
* **Visual Analytics:** Generates and saves visual outputs, including the "Eigenfaces" (principal components) and side-by-side reconstruction comparisons.

## Technical Details

1. **Language:**
   * Written in **Python**, utilizing **NumPy** for high-performance matrix operations and **Matplotlib** for visualization.
2. **Algorithm:**
   * Uses **Principal Component Analysis (PCA)** to calculate the eigenvectors (Eigenfaces) of the face dataset.
   * Projects data into a lower-dimensional subspace defined by the top K principal components.
3. **Math Strategies:**
   * **Covariance Method:** Direct diagonalization of the pixel-wise covariance matrix ($D \times D$).
   * **Dual Method:** Efficient diagonalization of the Gram matrix ($N \times N$) for high-resolution datasets.
   * **SVD:** Robust factorization ensuring numerical stability.
4. **Classification:**
   * Uses a **Nearest Neighbor** classifier based on L2 norm (Euclidean distance) in the reduced feature space.
   * Implements threshold-based rejection for unknown faces.

## License

This project is released under the **MIT License**. Feel free to use, modify, and distribute the code.

## Contact

For questions, suggestions, or contributions, please feel free to reach out via GitHub.
