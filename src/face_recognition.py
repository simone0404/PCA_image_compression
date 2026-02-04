import numpy as np
import matplotlib.pyplot as plt
import os


def recognition(weights_train, weights_image, train_dataset, image, img_name, W, mean, threshold=4000.0, target_size=(92, 112), output_folder="output"):
    distances = np.linalg.norm(weights_train - weights_image, axis=1)  # Distance in components
    min_dist_idx = np.argmin(distances)
    min_distance = distances[min_dist_idx]
    predicted_id = min_dist_idx // 9
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Risultato di riconoscimento, distanza: {min_distance:.2f}")
    if min_distance > threshold:
        print(f"{img_name} non riconosciuto")
        plt.suptitle(f"IMMAGINE NON RICONOSCIBILE O NON PRESENTE NEL DATASET \n distanza eccessiva: {min_distance:.2f} > {threshold}")
    plt.subplot(2, 2, 1)
    plt.imshow(image.reshape(target_size[::-1]), cmap='gray', interpolation="nearest", vmin=0, vmax=255)
    plt.title("Immagine test")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    match_img = train_dataset[min_dist_idx]
    plt.imshow(match_img.reshape(target_size[::-1]), cmap='gray', interpolation="nearest", vmin=0, vmax=255)
    plt.title(f"Id della persona: {predicted_id}")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    image_reconstructed = np.dot(weights_image, W.T) + mean
    plt.imshow(image_reconstructed.reshape(target_size[::-1]), cmap='gray', interpolation="nearest", vmin=0, vmax=255)
    plt.title(f"Immagine test ricostruita")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    train_reconstructed = np.dot(weights_train[min_dist_idx], W.T) + mean
    plt.imshow(train_reconstructed.reshape(target_size[::-1]), cmap='gray', interpolation="nearest", vmin=0, vmax=255)
    plt.title(f"Immagine dataset ricostruita")
    plt.axis('off')
    filename = os.path.join(output_folder, f"recog_test_{img_name}_pred_{predicted_id+1}.png")
    plt.savefig(filename)
    plt.close()
