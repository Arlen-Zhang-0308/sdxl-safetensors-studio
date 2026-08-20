# SDXL Safetensors Studio

面向 Windows 的本地 SDXL `.safetensors` 生图工作台，使用 Python 3.12、FastAPI、Diffusers 和 PyTorch。

## 功能

- 扫描 `models/` 中的 SDXL 单文件 `.safetensors`，刷新后加入下拉列表。
- 扫描同级 `loras/` 目录，支持选择 LoRA 和调整权重。
- 默认参数：832×1216、CFG 4、Steps 30、Euler a、Seed -1。
- 支持 Euler a、Euler、DPM++ 2M Karras、UniPC。
- 页面底部持久化生成历史；每张图左上角可一键回填全部参数。
- 启动时检测 CUDA，优先 NVIDIA CUDA；不可用时自动使用 CPU。
- 所有图片和参数只保存在本机 `data/history/`。

## Windows 安装

1. 安装 64 位 Python 3.12：

   ```powershell
   winget install Python.Python.3.12
   ```

2. 双击 `install_windows.bat`。脚本会直接向 Python 3.12 主环境安装依赖，不创建虚拟环境；优先安装 CUDA 版 PyTorch，若安装失败才回退 CPU 版。
3. 将 SDXL 模型放入 `models/`，例如：

   ```text
   models\my-sdxl-model.safetensors
   ```

4. 可选：将 SDXL LoRA 放入 `loras/`：

   ```text
   loras\detail-enhancer.safetensors
   ```

5. 双击 `start_windows.bat`。浏览器会打开 <http://127.0.0.1:7860>。
6. 点击模型区域的刷新按钮，选择模型后开始生成。

## 目录

```text
sdxl-safetensors-studio/
├─ app/                 # FastAPI 和推理引擎
├─ static/              # 本地 Web UI
├─ models/              # SDXL safetensors
├─ loras/               # LoRA safetensors
├─ data/history/        # PNG 与完整参数 JSON
├─ install_windows.bat
└─ start_windows.bat
```

## CUDA 说明

- 需要 NVIDIA 显卡及兼容驱动。
- 启动窗口会打印 `CUDA: True/False` 和 GPU 名称，页面右上角也会显示设备。
- 脚本默认安装 PyTorch CUDA 12.8 wheel。它包含所需 CUDA runtime，但仍需要足够新的 NVIDIA 驱动。
- CPU 回退可以运行，但 SDXL 生成通常非常慢。
- 832×1216 对显存有一定要求；显存不足时，请先降低宽高和批量数。

## 模型兼容性

后端通过 `StableDiffusionXLPipeline.from_single_file()` 加载 SDXL 单文件权重。请使用完整 SDXL checkpoint，而不是仅包含 UNet、VAE 或 LoRA 的文件。首次加载可能需要几十秒，并会占用大量内存/显存。

默认启用 `local_files_only=True`。模型若缺少 Diffusers 无法从单文件推断的配置，加载会报错而不会自动联网下载；请换用结构完整的 SDXL checkpoint。

## 开发测试

```powershell
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest -q
py -3.12 -m ruff check app tests
```

自动化测试不需要真实模型或 GPU；真实推理需在安装了模型的 Windows/NVIDIA 环境验收。
