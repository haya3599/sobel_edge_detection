import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

# -----------------------------
# Config
# -----------------------------
IMAGE_DIR = "images"  # שימי פה את 5 התמונות (jpg/png)
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# -----------------------------
# Helper functions
# -----------------------------
def load_grayscale_images_to_ndarray(image_dir: str):
    """
    Loads images from a folder, converts to grayscale, and returns:
    - images_arr: ndarray of shape (N, H, W), dtype float32 in range [0, 1]
    - filenames: list of filenames
    Note: assumes all images have the same size. If not, we resize to first image size.
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Folder '{image_dir}' not found. Create it and put 5 bird grayscale images inside.")

    files = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(VALID_EXT)]
    if len(files) < 5:
        raise ValueError(f"Need at least 5 images in '{image_dir}'. Found: {len(files)}")

    # Load first to set a target size (so everything becomes a clean ndarray)
    first_img = Image.open(os.path.join(image_dir, files[0])).convert("L")
    target_size = first_img.size  # (W, H)

    images = []
    for f in files[:5]:
        img = Image.open(os.path.join(image_dir, f)).convert("L")
        if img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]
        images.append(arr)

    images_arr = np.stack(images, axis=0)  # (N, H, W)
    return images_arr, files[:5]

def sobel_feature_maps(images_arr: np.ndarray):
    """
    For each image, computes:
    - gx feature map (Sobel-x)
    - gy feature map (Sobel-y)
    - magnitude map sqrt(gx^2 + gy^2)

    Returns gx_arr, gy_arr, mag_arr each shape (N, H, W)
    """
    # Sobel kernels (standard, correct)
    Gx = np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32)

    Gy = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32)

    gx_list, gy_list, mag_list = [], [], []

    for img in images_arr:
        gx = convolve2d(img, Gx, mode="same", boundary="symm")
        gy = convolve2d(img, Gy, mode="same", boundary="symm")
        mag = np.sqrt(gx**2 + gy**2)

        gx_list.append(gx.astype(np.float32))
        gy_list.append(gy.astype(np.float32))
        mag_list.append(mag.astype(np.float32))

    return np.stack(gx_list, axis=0), np.stack(gy_list, axis=0), np.stack(mag_list, axis=0)

def show_results(images_arr, gx_arr, gy_arr, mag_arr, filenames):
    """
    Displays per image: original, gx, gy, magnitude.
    """
    n = images_arr.shape[0]
    for i in range(n):
        fig = plt.figure(figsize=(12, 3))

        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(images_arr[i], cmap="gray")
        ax1.set_title(f"Original\n{filenames[i]}")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(gx_arr[i], cmap="gray")
        ax2.set_title("Sobel Gx (feature map)")
        ax2.axis("off")

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(gy_arr[i], cmap="gray")
        ax3.set_title("Sobel Gy (feature map)")
        ax3.axis("off")

        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(mag_arr[i], cmap="gray")
        ax4.set_title("Magnitude sqrt(Gx^2 + Gy^2)")
        ax4.axis("off")

        plt.tight_layout()
        plt.show()
# Run
images_arr, filenames = load_grayscale_images_to_ndarray(IMAGE_DIR)
print("Loaded ndarray shape:", images_arr.shape, "| dtype:", images_arr.dtype)

gx_arr, gy_arr, mag_arr = sobel_feature_maps(images_arr)

print("Gx feature maps shape:", gx_arr.shape)
print("Gy feature maps shape:", gy_arr.shape)
print("Magnitude maps shape:", mag_arr.shape)

show_results(images_arr, gx_arr, gy_arr, mag_arr, filenames)
