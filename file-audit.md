# 文件变更审计

时间：2026-08-20
项目：SDXL Safetensors Studio

| 文件/目录 | 类型 | 用途 |
|---|---|---|
| `app/__init__.py` | 新建 | 后端包入口 |
| `app/schemas.py` | 新建 | 生图参数模型与校验 |
| `app/storage.py` | 新建 | 模型扫描、路径安全、历史存储 |
| `app/engine.py` | 新建 | CUDA 检测、SDXL、LoRA 与采样器推理 |
| `app/main.py` | 新建 | FastAPI 页面和 API 服务 |
| `static/index.html` | 新建 | 深灰工业工作台页面 |
| `static/styles.css` | 新建 | 响应式、状态与可访问样式 |
| `static/app.js` | 新建 | 扫描、生成、历史和参数回填交互 |
| `tests/test_storage.py` | 新建 | 扫描、路径与历史测试 |
| `tests/test_api.py` | 新建 | API、参数和历史落盘测试 |
| `models/` | 新建 | SDXL safetensors 专用目录 |
| `loras/` | 新建 | LoRA safetensors 专用目录 |
| `data/history/` | 新建 | 本地图片与参数历史目录 |
| `install_windows.bat` | 新建 | Python 3.12 / CUDA PyTorch 安装 |
| `start_windows.bat` | 新建 | 设备检测与一键启动 |
| `pyproject.toml` | 新建 | Python 3.12 项目元数据 |
| `requirements.txt` | 新建 | 运行依赖 |
| `README.md` | 新建 | Windows 使用与故障说明 |
| `.gitignore` | 新建 | 排除权重、历史和虚拟环境 |
| `LICENSE` | 新建 | MIT 许可证 |

## 2026-08-20 主环境运行方式调整

| 文件 | 类型 | 变更 |
|---|---|---|
| `install_windows.bat` | 修改 | 移除 `.venv` 创建和激活，改用 `py -3.12 -m pip` 直接向主环境安装依赖 |
| `start_windows.bat` | 修改 | 移除虚拟环境检查和激活，改用 `py -3.12` 检测设备并启动 Uvicorn |
| `README.md` | 修改 | 安装、启动和开发测试说明统一改为 Python 3.12 主环境 |
| `.gitignore` | 修改 | 移除不再使用的 `.venv/` 忽略项 |

## 2026-08-20 本地 Git 初始版本

| 文件 | 类型 | 变更 |
|---|---|---|
| `.gitignore` | 修改 | 排除交付用 `sdxl-safetensors-studio.zip`，避免将构建产物纳入初始提交 |
| 项目 Git 仓库 | 初始化 | 以当前项目源文件作为初始版本，本地提交身份使用 `arlen` |

## 2026-08-20 推理进度条

| 文件 | 类型 | 变更 |
|---|---|---|
| `app/tasks.py` | 新建 | 提供线程安全的内存任务状态和进度快照 |
| `app/engine.py` | 修改 | 接入 Diffusers `callback_on_step_end`，逐采样 step 上报真实进度 |
| `app/main.py` | 修改 | 生图接口改为后台任务，新增任务状态查询 API |
| `static/index.html` | 修改 | 在推理状态区增加语义化进度条、百分比和 step 文本 |
| `static/styles.css` | 修改 | 增加琥珀色推理进度轨道及状态布局 |
| `static/app.js` | 修改 | 创建任务后轮询进度，完成后显示图片并刷新历史 |
| `tests/test_api.py` | 修改 | 覆盖任务创建、状态查询、100% 完成和未知任务 |
| `tests/test_tasks.py` | 新建 | 覆盖任务状态更新和快照隔离 |
