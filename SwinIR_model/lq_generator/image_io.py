# image_io.py

import os
import glob
import torch
import cv2
import numpy as np


SUPPORTED_EXT = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.PNG", "*.JPG", "*.JPEG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP"]


# ==========================================================
# 列出所有图片路径
# ==========================================================

def list_images(input_dir, recursive=True):

    files = []

    if recursive:
        for ext in SUPPORTED_EXT:
            files.extend(
                glob.glob(
                    os.path.join(input_dir, "**", ext),
                    recursive=True
                )
            )
    else:
        for ext in SUPPORTED_EXT:
            files.extend(
                glob.glob(
                    os.path.join(input_dir, ext)
                )
            )

    files.sort()
    return files


# ==========================================================
# 读取图像 -> torch tensor
# ==========================================================

def load_image(path):

    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    return img


# ==========================================================
# 保存 tensor 为 PNG
# ==========================================================

def save_image(tensor, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    img = np.clip(img, 0, 1)
    img = (img * 255.0).round().astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, img)
