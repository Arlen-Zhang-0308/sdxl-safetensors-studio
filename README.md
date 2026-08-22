# SDXL Safetensors Studio

面向 Windows 的本地 AI 生图工作台，支持 SDXL 单文件及 Diffusers 目录模型（包括 SD、SDXL 和 Z-Image），使用 Python 3.12、FastAPI、Diffusers 和 PyTorch。

## 功能

- 扫描 `models/` 中的 SDXL 单文件 `.safetensors` 和含 `model_index.json` 的 Diffusers 模型目录，刷新后加入下拉列表。
- 扫描同级 `loras/` 目录，支持选择 LoRA 和调整权重。
- 扫描 `ip_adapters/` 中的本地 IP-Adapter 包，支持图像提示和 0–2 影响强度。
- 默认参数：832×1216、CFG 4、Steps 30、Euler a、Seed -1。
- 默认 Clip Skip 为 2；支持 0–12，0 表示使用 Diffusers 默认文本编码层。
- 超过 SDXL CLIP 上限的提示词会自动按 token 分块编码，并平均 sequence/pooled embeddings，避免直接截断。
- 支持 21 个采样配置：Euler、Heun、LMS、DDIM、PNDM、DPM2、DPM++、DEIS、UniPC 及常用 Karras/SDE 变体。
- 页面底部持久化生成历史；每张图左上角可一键回填全部参数。
- 支持 SDXL 图生图：上传 PNG/JPEG/WebP 参考图、预览并调整重绘强度。
- 支持局部重绘：直接在参考图上用画笔/橡皮绘制蒙版，仅重绘红色覆盖区域。
- 启动时检测 CUDA，优先 NVIDIA CUDA；不可用时自动使用 CPU。
- 所有图片和参数只保存在本机 `data/history/`。
- 图生图参考图会转换为 PNG 并保存在本机 `data/inputs/`，单张最大 20MB。

## Windows 安装

1. 安装 64 位 Python 3.12：

   ```powershell
   winget install Python.Python.3.12
   ```

2. 双击 `install_windows.bat`。脚本会直接向 Python 3.12 主环境安装依赖，不创建虚拟环境；优先安装 CUDA 版 PyTorch，若安装失败才回退 CPU 版。
3. 将模型放入 `models/`。支持 SDXL 单文件：

   ```text
   models\my-sdxl-model.safetensors
   ```

   也支持完整 Diffusers 目录模型：

   ```text
   models\my-diffusers-model\
   ├─ model_index.json
   ├─ scheduler\
   ├─ text_encoder\
   ├─ tokenizer\
   ├─ unet\
   └─ vae\
   ```

4. 可选：将 SDXL LoRA 放入 `loras/`：

   ```text
   loras\detail-enhancer.safetensors
   ```

5. 可选：将一个完整 IP-Adapter 包放入独立子目录：

   ```text
   ip_adapters\sdxl-plus\
   ├─ ip-adapter-plus_sdxl_vit-h.safetensors
   └─ image_encoder\
      ├─ config.json
      └─ model.safetensors
   ```

6. 双击 `start_windows.bat`。浏览器会打开 <http://127.0.0.1:7860>。
7. 点击模型区域的刷新按钮，选择模型后开始生成。

### IP-Adapter

1. 在模型配置中选择扫描到的 IP-Adapter 包。
2. 上传 PNG、JPEG 或 WebP 图像提示。
3. 调整“图像影响强度”；默认 0.60，越高越强调参考图特征。
4. IP-Adapter 可与文生图、图生图、蒙版局部重绘和 LoRA 同时使用。

程序强制 `local_files_only=True`，不会在生成时联网下载权重。适配器权重必须与 SDXL 兼容；包目录必须同时包含根目录 `.safetensors` 权重和完整 `image_encoder/`。

Diffusers 0.35 的本地适配器加载需要显式指定包根目录；当前版本已按本地目录模式传入。若推理失败，页面会显示真实异常类型和消息，便于区分文件缺失、显存不足和版本兼容问题。

### 图生图

1. 在“生成模式”中选择“图生图”。
2. 上传 PNG、JPEG 或 WebP 参考图。
3. 调整“重绘强度”：数值越低越接近参考图，越高改动越大。
4. 可选：在“局部重绘蒙版”中涂抹需要修改的区域；红色区域会重绘，透明区域保持原图。
5. 填写提示词并生成。历史记录的“应用参数”会恢复模式、参考图、蒙版和强度。

未绘制蒙版时仍执行普通图生图；绘制蒙版后会根据模型自动切换到对应的 Inpaint 管线。蒙版会以黑底白区 PNG 保存在 `data/inputs/`。

## 目录

```text
sdxl-safetensors-studio/
├─ app/                 # FastAPI 和推理引擎
├─ static/              # 本地 Web UI
├─ models/              # SDXL safetensors / Diffusers 模型目录
├─ loras/               # LoRA safetensors
├─ ip_adapters/         # 本地 IP-Adapter 包目录
├─ data/history/        # PNG 与完整参数 JSON
├─ data/inputs/         # 图生图参考图
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

后端通过 `StableDiffusionXLPipeline.from_single_file()` 加载 SDXL 单文件权重；对于含 `model_index.json` 的模型目录，通过 `DiffusionPipeline.from_pretrained()` 按目录配置自动选择 SD、SDXL 或 Z-Image 管线。目录必须包含配置引用的完整组件，不能只有 UNet、VAE、Transformer 或 checkpoint 文件。首次加载可能需要几十秒，并会占用大量内存/显存。

目录模型只扫描 `models/` 的直接子目录；`.cache` 等不含 `model_index.json` 的目录会被忽略。截图所示仅有一个 `text_encoder` 与 `tokenizer`、并带 `v1-5-pruned.ckpt` 的布局通常属于 Stable Diffusion 1.5，而不是 SDXL；当前版本可以加载这种完整 Diffusers 目录，但 SDXL 专用 IP-Adapter 不可与 SD 1.5 模型混用。

默认启用 `local_files_only=True`。模型若缺少本地配置或组件，加载会报错而不会自动联网下载；请换用结构完整的 checkpoint 或 Diffusers 目录。

不同采样器对模型和步数的表现不同。Karras、SDE 及 DPM++ 变体并不保证对所有自定义 checkpoint 都优于 Euler a；建议先固定 seed 对比。

Z-Image 使用专用 Transformer 架构：CUDA 下按官方建议使用 BF16，并保留模型自带的 FlowMatch 调度器，因此页面选择的传统 SD 采样器不会覆盖它。Z-Image 不接收 Clip Skip；当前 SDXL IP-Adapter 也不能用于 Z-Image，若误选会返回明确的兼容性错误。Z-Image 需要 Diffusers 0.36 或更高版本，更新项目后请重新运行 `install_windows.bat`。

## 开发测试

```powershell
py -3.12 -m pip install -e ".[dev]"
py -3.12 -m pytest -q
py -3.12 -m ruff check app tests
```

自动化测试不需要真实模型或 GPU；真实推理需在安装了模型的 Windows/NVIDIA 环境验收。
