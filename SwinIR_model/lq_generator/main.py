# main.py

from pathlib import Path

import yaml
import torch

from degradation_pipeline import RealESRGANDegradation
from batch_runner import BatchProcessor

BASE_DIR = Path(__file__).resolve().parent


# ==========================================================
# 读取配置
# ==========================================================

def load_config(config_path):

    with open(config_path, "r", encoding='utf-8') as f:
        opt = yaml.safe_load(f)

    return opt


def _resolve_from_base(base_dir, raw_path):
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


# ==========================================================
# 主函数
# ==========================================================

def main():

    config_path = (BASE_DIR / "degradation_config.yml").resolve()

    opt = load_config(config_path)
    config_dir = config_path.parent

    if "io" in opt:
        for key in ("input_dir", "output_dir"):
            if key in opt["io"] and opt["io"][key]:
                opt["io"][key] = _resolve_from_base(config_dir, opt["io"][key])

    if "log" in opt and "log_path" in opt["log"] and opt["log"]["log_path"]:
        opt["log"]["log_path"] = _resolve_from_base(config_dir, opt["log"]["log_path"])

    if opt["manual_seed"] is not None:
        torch.manual_seed(opt["manual_seed"])

    device = opt["device"] if torch.cuda.is_available() else "cpu"

    pipeline = RealESRGANDegradation(opt, device=device)

    processor = BatchProcessor(opt, pipeline)

    processor.run()


if __name__ == "__main__":
    main()
