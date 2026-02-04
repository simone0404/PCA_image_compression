import matplotlib.pyplot as plt
import os
import numpy as np
from src.decorators import ensure_folder


@ensure_folder
def eigen_faces(W, target_size=(92, 112), output_folder="output"):
    """
    Plot the first 20 eigenfaces as images
    Target size is (width, height)
    """
    components = W.shape[1]
    plt.figure(figsize=(15, 10))
    for i in range(min(components, 20)):
        plt.subplot(4, 5, i + 1)
        eigenface = W[:, i].reshape(target_size[::-1])
        plt.imshow(eigenface, cmap="grey", interpolation="nearest")
        plt.title(f"Eigenface {i+1}")
        plt.axis("off")
    plt.savefig(os.path.join(output_folder, "eigenfaces.png"))
    plt.close()


@ensure_folder
def comparison(original, reconstructed, img_name, target_size=(92, 112), output_folder="output"):
    """
    Plot original and reconstructed images
    Target size is (width, height)
    """
    mse = np.mean((original - reconstructed) ** 2)  # sigma_noise^2
    variance = np.var(original)  # sigma_signal^2
    nsr = mse / variance if variance > 0 else 0  # noise-signal ratio
    similarity = (1 - nsr) * 100 if variance > 0 else 0
    plt.figure(figsize=(8, 4))
    plt.suptitle(f"MSE = {mse:.4f}, Qualità = {similarity:.2f}%")
    plt.subplot(1, 2, 1)
    plt.imshow(original.reshape(target_size[::-1]), cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    plt.title("Originale")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.reshape(target_size[::-1]), cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    plt.title("Ricostruita")
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"reconstruction_{img_name}.png"))
    plt.close()
