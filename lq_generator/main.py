# main.py

import yaml
import torch

from rename import rename_images
from degradation_pipeline import RealESRGANDegradation
from batch_runner import BatchProcessor


# ==========================================================
# 读取配置
# ==========================================================

def load_config(config_path):

    with open(config_path, "r", encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    return opt


# ==========================================================
# 主函数
# ==========================================================

def main():

    config_path = "degradation_config.yml"

    opt = load_config(config_path)

    if opt["manual_seed"] is not None:
        torch.manual_seed(opt["manual_seed"])

    device = opt["device"] if torch.cuda.is_available() else "cpu"

    pipeline = RealESRGANDegradation(opt, device=device)

    processor = BatchProcessor(opt, pipeline)

    processor.run()


if __name__ == "__main__":
    main()
