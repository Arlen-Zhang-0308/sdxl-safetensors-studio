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

## 2026-08-20 图生图功能

| 文件 | 类型 | 变更 |
|---|---|---|
| `app/schemas.py` | 修改 | 增加文生图/图生图模式、参考图与重绘强度校验 |
| `app/storage.py` | 修改 | 增加参考图目录、安全路径解析及历史参考图 URL |
| `app/engine.py` | 修改 | 从 SDXL 组件派生 Img2Img 管线，支持参考图与 strength |
| `app/main.py` | 修改 | 增加图片上传 API、参考图缩放和图生图后台任务 |
| `static/index.html` | 修改 | 增加模式切换、参考图预览、移除和重绘强度控件 |
| `static/styles.css` | 修改 | 增加图生图模式及参考图控件样式 |
| `static/app.js` | 修改 | 增加上传、校验、图生图请求和历史完整恢复 |
| `tests/test_api.py` | 修改 | 覆盖上传、格式拒绝、图生图生成与历史数据 |
| `data/inputs/.gitkeep` | 新建 | 保留本地参考图目录 |
| `.gitignore` | 修改 | 排除用户上传的参考图文件 |

## 2026-08-20 历史图片删除

| 文件 | 类型 | 变更 |
|---|---|---|
| `app/storage.py` | 修改 | 增加历史 ID 校验及 PNG/JSON 成对删除，不删除参考图 |
| `app/main.py` | 修改 | 增加单条历史记录 DELETE API |
| `static/app.js` | 修改 | 每张历史卡片增加删除确认、请求和即时刷新 |
| `static/styles.css` | 修改 | 增加右上角危险操作按钮及禁用状态 |
| `tests/test_storage.py` | 修改 | 覆盖安全删除、参考图保留与非法 ID |
| `tests/test_api.py` | 修改 | 覆盖删除成功、重复删除和非法 ID 响应 |

## 2026-08-20 Clip Skip 与长提示词分块编码

| 文件 | 类型 | 变更 |
|---|---|---|
| `app/prompt_encoding.py` | 新建 | 实现 SDXL 双 tokenizer 长提示词分块编码及 embeddings 平均 |
| `app/schemas.py` | 修改 | 增加 0–12 的 Clip Skip 参数，默认 2 |
| `app/engine.py` | 修改 | 文生图和图生图统一接入 Clip Skip 与分块编码参数 |
| `app/main.py` | 修改 | 状态默认值增加 Clip Skip=2 |
| `static/index.html` | 修改 | 生成参数区增加 Clip Skip 输入 |
| `static/app.js` | 修改 | 提交、历史展示与参数回填支持 Clip Skip |
| `tests/test_prompt_encoding.py` | 新建 | 覆盖短提示原生路径、长提示分块平均和 Clip Skip 透传 |
| `tests/test_api.py` | 修改 | 覆盖默认值、历史持久化和范围校验 |
| `README.md` | 修改 | 说明 Clip Skip 与长提示词处理规则 |

## 2026-08-21 图生图局部重绘蒙版

| 文件 | 类型 | 变更 |
|---|---|---|
| `app/schemas.py` | 修改 | 增加可选蒙版文件参数及模式校验 |
| `app/storage.py` | 修改 | 历史记录增加蒙版图片 URL |
| `app/engine.py` | 修改 | 有蒙版时由 SDXL 组件派生 Inpaint 管线并传入 mask_image |
| `app/main.py` | 修改 | 加载、灰度化和缩放蒙版并接入后台生成任务 |
| `static/index.html` | 修改 | 增加蒙版画布、画笔、橡皮、清空和笔刷大小控件 |
| `static/styles.css` | 修改 | 增加局部重绘编辑器和双层画布样式 |
| `static/app.js` | 修改 | 支持鼠标/触控绘制、蒙版导出上传、历史恢复和局部重绘请求 |
| `tests/test_api.py` | 修改 | 覆盖蒙版上传、灰度缩放、历史记录和模式限制 |
| `README.md` | 修改 | 增加局部重绘操作与存储说明 |
| `.musa-manifest.json` | 修改 | 更新生成时间并保持文件目录清单同步 |
| `FILES.txt` | 修改 | 重新生成项目目录清单 |

## 2026-08-22 Diffusers 目录模型支持

| 文件 | 类型 | 变更 |
|---|---|---|
| `app/storage.py` | 修改 | 扫描 `models/` 下含 `model_index.json` 的直接子目录，并安全解析文件或目录模型路径 |
| `app/engine.py` | 修改 | 目录模型改用离线 `DiffusionPipeline.from_pretrained()` 加载，并按单/双文本编码器选择 SD 或 SDXL 图生图、局部重绘管线 |
| `app/prompt_encoding.py` | 修改 | 对单文本编码器模型使用原生提示词参数，保留 SDXL 双编码器长提示词分块逻辑 |
| `tests/test_storage.py` | 修改 | 覆盖目录模型扫描、合法解析、不完整目录忽略及路径穿越拒绝 |
| `tests/test_engine.py` | 修改 | 覆盖目录模型通过 `from_pretrained()` 离线加载及 CPU 放置 |
| `README.md` | 修改 | 增加目录结构示例、SD/SDXL 自动识别及适配器兼容性说明 |
| `.musa-manifest.json` | 修改 | 更新生成时间并保持文件目录清单同步 |
| `FILES.txt` | 检查 | 本次无新增或删除文件，目录清单内容保持不变 |

## 2026-08-21 IP-Adapter 本地加载修复

| 文件 | 类型 | 变更 |
|---|---|---|
| `app/engine.py` | 修改 | 为 Diffusers 0.35 本地 IP-Adapter 加载显式传入空 subfolder，修复缺少必选参数导致的 TypeError |
| `app/main.py` | 修改 | 非 RuntimeError 失败时返回真实异常类型和消息，不再误报为 SDXL checkpoint 不兼容 |
| `tests/test_engine.py` | 新建 | 锁定本地适配器路径、subfolder、编码器目录和离线参数 |
| `tests/test_api.py` | 修改 | 覆盖后台任务真实异常消息透传 |
| `README.md` | 修改 | 增加本地加载兼容修复和错误诊断说明 |
| `.musa-manifest.json` | 修改 | 更新生成时间并保持文件目录清单同步 |
| `FILES.txt` | 修改 | 重新生成项目目录清单 |

## 2026-08-21 IP-Adapter 与扩展采样器

| 文件 | 类型 | 变更 |
|---|---|---|
| `app/schemas.py` | 修改 | 增加 IP-Adapter 包、图像提示和 0–2 强度参数及成对校验 |
| `app/storage.py` | 修改 | 扫描完整本地适配器包、安全解析权重并返回历史图像提示 URL |
| `app/engine.py` | 修改 | 支持离线加载/切换 IP-Adapter，并扩展 21 个 SDXL 采样配置 |
| `app/main.py` | 修改 | 接入适配器包、图像提示与采样器状态和后台任务 |
| `static/index.html` | 修改 | 增加 IP-Adapter 选择、图像上传预览和强度控件 |
| `static/app.js` | 修改 | 增加适配器扫描、上传、提交、校验和历史参数恢复 |
| `tests/test_api.py` | 修改 | 覆盖适配器生成、历史记录、成对校验及采样器数量 |
| `tests/test_storage.py` | 修改 | 覆盖完整包扫描、不完整包忽略和安全路径 |
| `ip_adapters/.gitkeep` | 新建 | 保留本地 IP-Adapter 包目录 |
| `.gitignore` | 修改 | 排除本地适配器权重与图像编码器文件 |
| `install_windows.bat` | 修改 | 安装完成提示增加适配器目录说明 |
| `README.md` | 修改 | 增加适配器包结构、操作说明和采样器说明 |
| `.musa-manifest.json` | 修改 | 更新生成时间并保持文件目录清单同步 |
| `FILES.txt` | 修改 | 重新生成项目目录清单 |
