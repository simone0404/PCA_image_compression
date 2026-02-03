import os
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener


register_heif_opener()


def load_dataset(root_folder, target_size=(92, 112)):
    """
    Extract images from root_folder and return them as a numpy array
    Only images with a ratio of at least target_size are kept
    Target size is (width, height)"""
    faces = []  # photo_number x pixel_number
    valid_extensions = ('.pgm', ".png", ".jpg", ".jpeg", ".heic")
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                path = os.path.join(subdir, file)
                img = Image.open(path)
                img = img.convert('L')
                w, h = img.size
                ideal_h = w * target_size[1] / target_size[0]
                if h < ideal_h-0.1:
                    continue
                if h > ideal_h+0.1:
                    img = img.crop((0, 0, w, int(ideal_h)))
                img = img.resize(target_size)
                img_matrix = np.asarray(img)  # convert to numpy ndarray
                img_vector = img_matrix.flatten()  # PCA needs vectors
                faces.append(img_vector)
    if not faces:
        print("No valid images found in the dataset directory")
    dataset = np.array(faces)
    return dataset
