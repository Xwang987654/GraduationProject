import torch
import torch.nn.functional as F
import random
import numpy as np
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

from basicsr.utils.img_process_util import filter2D
from basicsr.utils import DiffJPEG, USMSharp

from utils import generate_kernel

class RealESRGANDegradation:

    def __init__(self, opt: dict, device="cuda"):

        self.opt = opt
        self.device = device
        self.scale = opt["scale"]

        self.jpeg = DiffJPEG(differentiable=False).to(device)

        # 初始化 blur kernel
        self.kernel1 = generate_kernel(opt).to(device)
        self.kernel2 = generate_kernel(opt).to(device)
        self.sinc_kernel = generate_kernel(opt, sinc=True).to(device)

    # ==========================================================
    # 主退化函数
    # ==========================================================
    @torch.no_grad()
    def degrade(self, gt):
        """
        对单张（或单batch）高清图像（GT）应用 Real-ESRGAN 风格的退化流程，
        生成对应的低质量图像（LQ）。

        Args:
            gt (torch.Tensor): 输入的高清图像张量。
                               形状: (1, C, H, W)
                               数值范围: [0, 1] (float)
                               通常 C=3 (RGB)

        Returns:
            torch.Tensor: 退化后的低质量图像张量。
                          形状: (1, C, H/scale, W/scale)
                          数值范围: [0, 1] (float)
        """

        # 将输入张量移动到指定设备（GPU/CPU）
        gt = gt.to(self.device)

        # 记录原始图像的高和宽，用于最后一步的固定下采样（按照 scale 比例）
        ori_h, ori_w = gt.shape[2:]

        # ================== 第一阶段退化 ==================
        # 对应 Real-ESRGAN 中的 first-order degradation：
        # 模糊 → 随机缩放 → 噪声 → JPEG压缩

        # ---- 1. 应用第一个模糊核（通常为各向异性/各向同性高斯模糊）----
        out = filter2D(gt, self.kernel1)

        # ---- 2. 随机缩放（上采样/下采样/保持原尺寸）----
        # 从配置中读取缩放概率分布和缩放范围
        resize_prob = self.opt["degradation_1"]["resize_prob"]
        resize_range = self.opt["degradation_1"]["resize_range"]

        # 根据概率随机选择操作类型：'up' 放大, 'down' 缩小, 'keep' 不变
        updown_type = random.choices(["up", "down", "keep"], resize_prob)[0]

        # 根据操作类型生成缩放因子 scale
        if updown_type == "up":
            scale = np.random.uniform(1, resize_range[1])  # 放大 (1, max]
        elif updown_type == "down":
            scale = np.random.uniform(resize_range[0], 1)  # 缩小 [min, 1)
        else:
            scale = 1  # 不变

        # 随机选择插值方式：区域自适应、双线性、双三次
        mode = random.choice(["area", "bilinear", "bicubic"])
        # 执行缩放
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # ---- 3. 添加噪声（高斯噪声 或 泊松噪声）----
        # 根据高斯噪声概率决定是否使用高斯噪声；否则使用泊松噪声
        if random.random() < self.opt["degradation_1"]["gaussian_noise_prob"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt["degradation_1"]["noise_range"],
                clip=True  # 将像素值裁剪到 [0,1]
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt["degradation_1"]["poisson_scale_range"],
                clip=True
            )

        # ---- 4. JPEG 压缩（可微分近似）----
        jpeg_range = self.opt["degradation_1"]["jpeg_range"]
        # 为 batch 中的每张图随机生成一个 JPEG 质量因子
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
        # 确保像素值在有效范围内
        out = torch.clamp(out, 0, 1)
        # 应用可微分 JPEG 压缩
        out = self.jpeg(out, quality=jpeg_p)

        # ================== 第二阶段退化 ==================
        # 对应 Real-ESRGAN 中的 second-order degradation：
        # 对第一阶段的结果再次应用类似的退化过程，增强退化程度

        # ---- 1. 以一定概率应用第二个模糊核（二次模糊）----
        if random.random() < self.opt["degradation_2"]["second_blur_prob"]:
            out = filter2D(out, self.kernel2)

        # ---- 2. 第二次随机缩放（参数来自第二阶段配置）----
        resize_prob2 = self.opt["degradation_2"]["resize_prob"]
        resize_range2 = self.opt["degradation_2"]["resize_range"]

        updown_type = random.choices(["up", "down", "keep"], resize_prob2)[0]

        if updown_type == "up":
            scale = np.random.uniform(1, resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(resize_range2[0], 1)
        else:
            scale = 1

        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # ---- 3. 第二次添加噪声（高斯/泊松）----
        if random.random() < self.opt["degradation_2"]["gaussian_noise_prob"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt["degradation_2"]["noise_range"],
                clip=True
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt["degradation_2"]["poisson_scale_range"],
                clip=True
            )

        # ---- 4. 最终下采样：固定将图像缩小到原始尺寸的 1/scale ----
        out = F.interpolate(
            out,
            size=(ori_h // self.scale, ori_w // self.scale),
            mode="bicubic"
        )

        # ---- 5. 应用 sinc 模糊核（模拟振铃效应）----
        out = filter2D(out, self.sinc_kernel)

        # ---- 6. 第二次 JPEG 压缩（最终压缩）----
        jpeg_range2 = self.opt["degradation_2"]["jpeg_range"]
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = self.jpeg(out, quality=jpeg_p)

        # ================== 注意：此处存在重复的第二阶段代码 ==================
        # 以下代码块与上面的第二阶段内容完全一致，可能是复制粘贴导致的冗余。
        # 在实际使用中应删除重复部分，避免两次执行相同的操作。
        # 但此处保留原代码，并添加注释说明。

        if random.random() < self.opt["degradation_2"]["second_blur_prob"]:
            out = filter2D(out, self.kernel2)

        resize_prob2 = self.opt["degradation_2"]["resize_prob"]
        resize_range2 = self.opt["degradation_2"]["resize_range"]

        updown_type = random.choices(["up", "down", "keep"], resize_prob2)[0]

        if updown_type == "up":
            scale = np.random.uniform(1, resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(resize_range2[0], 1)
        else:
            scale = 1

        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # noise2
        if random.random() < self.opt["degradation_2"]["gaussian_noise_prob"]:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.opt["degradation_2"]["noise_range"],
                clip=True
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt["degradation_2"]["poisson_scale_range"],
                clip=True
            )

        # 最终下采样到 scale
        out = F.interpolate(
            out,
            size=(ori_h // self.scale, ori_w // self.scale),
            mode="bicubic"
        )

        # 最终 sinc + jpeg
        out = filter2D(out, self.sinc_kernel)

        jpeg_range2 = self.opt["degradation_2"]["jpeg_range"]
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range2)
        out = torch.clamp(out, 0, 1)
        out = self.jpeg(out, quality=jpeg_p)

        # ==============================================================

        # 最终输出裁剪到 [0,1] 范围，确保像素值合法
        return torch.clamp(out, 0, 1)
