# SwinIR_UI

本目录提供一个独立的本地 Web UI，用于展示 `GraduationProject` 里的 SwinIR 推理效果，包含：

- 单图演示：上传 1 张图，处理后直接前后对比。
- 批量处理：选择一个文件夹上传，批量处理所有图片并保存到结果目录。

## 目录结构

- `app.py`: Flask 后端，负责模型加载与推理接口。
- `templates/index.html`: 页面结构。
- `static/styles.css`: 页面视觉样式。
- `static/app.js`: 前端交互逻辑。
- `results/`: 处理结果输出目录（运行后自动生成子目录）。

## 启动方式

在 `Z:\Code_Pile\毕业设计\SwinIR-main\GraduationProject\SwinIR_UI` 下执行：

```bash
pip install -r requirements.txt
python app.py
```

启动后访问：

```text
http://127.0.0.1:7860
```

## 功能说明

1. 单图模式
   - 点击“选择图片”或拖拽图片。
   - 点击“开始处理”后调用 `/api/process-single`。
   - 页面会显示处理后图片，并支持滑块查看前后差异。

2. 批量模式
   - 点击“选择文件夹”。
   - 浏览器会上传文件夹中的所有文件（推荐 Chrome/Edge）。
   - 点击“开始批量处理”调用 `/api/process-batch`，输出写入 `results/batch/<时间戳_随机ID>/`。

## 模型来源

后端复用了 `SwinIR_model/main_test_swinir.py` 中的模型结构定义和 tile 推理逻辑，当前提供以下模型选项：

- Real-SR x2 (PSNR)
- Real-SR x2 (GAN)
- Real-SR x4 (PSNR)

对应权重默认读取：

- `SwinIR_model/model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_PSNR-with-dict-keys-params-and-params_ema.pth`
- `SwinIR_model/model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN-with-dict-keys-params-and-params_ema.pth`
- `SwinIR_model/model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_PSNR-with-dict-keys-params-and-params_ema.pth`
