import matplotlib.pyplot as plt
import numpy as np
import os


def eigen_faces(W, h, w, output_folder="output"):
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)
    n_components = W.shape[1]
    plt.figure(figsize=(15, 10))
    for i in range(min(n_components,20)):
        plt.subplot(4, 5, i + 1)
        eigenface = W[:, i].reshape((h, w))
        plt.imshow(eigenface, cmap="gray")
        plt.title(f"Eigenface {i+1}")
        plt.axis("off")
    plt.savefig(os.path.join(output_folder, "eigenfaces.png"))
    plt.close()


def comparison(original, reconstructed, h, w, img_index, output_folder="output"):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original.reshape(h, w), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.reshape(h, w), cmap='gray')
    plt.axis('off')
    plt.savefig(f"{output_folder}/reconstruction_{img_index}.png")
    plt.close()
