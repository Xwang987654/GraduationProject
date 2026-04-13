# batch_runner.py

import os

import torch
from tqdm import tqdm

from image_io import list_images, load_image, save_image


class BatchProcessor:

    def __init__(self, opt, pipeline):

        self.opt = opt
        self.pipeline = pipeline

        self.input_dir = opt["io"]["input_dir"]
        self.output_dir = opt["io"]["output_dir"]
        self.recursive = opt["io"]["recursive"]
        self.overwrite = opt["io"]["overwrite"]

    # ==========================================================
    # 构建输出路径
    # ==========================================================

    def _build_save_path(self, img_path):

        rel_path = os.path.relpath(img_path, self.input_dir)
        rel_path = os.path.splitext(rel_path)[0] + ".png"

        save_path = os.path.join(self.output_dir, rel_path)
        return save_path

    # ==========================================================
    # 主运行函数
    # ==========================================================

    def run(self):

        image_list = list_images(self.input_dir, self.recursive)

        if len(image_list) == 0:
            raise ValueError("No images found in input directory.")

        print(f"Found {len(image_list)} images.")

        for img_path in tqdm(image_list):

            save_path = self._build_save_path(img_path)

            if (not self.overwrite) and os.path.exists(save_path):
                continue

            gt = load_image(img_path)

            lq = self.pipeline.degrade(gt)

            # 每次处理完一战图片就清除gpu缓存避免OOM
            torch.cuda.empty_cache()

            save_image(lq, save_path)

        print("LQ generation completed.")
