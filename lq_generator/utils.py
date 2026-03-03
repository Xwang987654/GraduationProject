# utils.py

import numpy as np
import torch
import math
import random


# ==========================================================
# 生成二维高斯 kernel
# ==========================================================

def _gaussian_kernel(kernel_size, sigma_x, sigma_y=None, rotation=0):

    if sigma_y is None:
        sigma_y = sigma_x

    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    # rotation
    theta = rotation
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)

    kernel = np.exp(
        -(xr**2 / (2 * sigma_x**2) + yr**2 / (2 * sigma_y**2))
    )

    kernel = kernel / np.sum(kernel)
    return kernel


# ==========================================================
# 生成 sinc kernel
# ==========================================================

def _sinc_kernel(kernel_size, cutoff):

    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    r = np.sqrt(xx**2 + yy**2)
    kernel = cutoff * np.sinc(cutoff * r)

    kernel = kernel / np.sum(kernel)
    return kernel


# ==========================================================
# 主接口：生成 kernel
# ==========================================================

def generate_kernel(opt, sinc=False):

    blur_opt = opt["blur"]

    kernel_size = blur_opt["kernel_size"]

    if sinc:
        cutoff = random.uniform(0.1, 1.0)
        kernel = _sinc_kernel(kernel_size, cutoff)
    else:
        kernel_type = random.choices(
            blur_opt["kernel_list"],
            blur_opt["kernel_prob"]
        )[0]

        if kernel_type == "iso":
            sigma = random.uniform(
                blur_opt["blur_sigma"][0],
                blur_opt["blur_sigma"][1]
            )
            kernel = _gaussian_kernel(kernel_size, sigma)

        elif kernel_type == "aniso":
            sigma_x = random.uniform(
                blur_opt["blur_sigma"][0],
                blur_opt["blur_sigma"][1]
            )
            sigma_y = random.uniform(
                blur_opt["blur_sigma"][0],
                blur_opt["blur_sigma"][1]
            )
            rotation = random.uniform(0, math.pi)
            kernel = _gaussian_kernel(
                kernel_size,
                sigma_x,
                sigma_y,
                rotation
            )
        else:
            raise ValueError("Unsupported kernel type")

    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)

    return kernel
