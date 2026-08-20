const $ = (selector) => document.querySelector(selector);
const form = $("#generateForm");
const modelSelect = $("#model");
const loraSelect = $("#lora");
const ipAdapterSelect = $("#ipAdapter");
const refreshButton = $("#refreshWeights");
const generateButton = $("#generateButton");
const previewGrid = $("#previewGrid");
const emptyState = $("#emptyState");
const loadingState = $("#loadingState");
const historyGrid = $("#historyGrid");
const historyEmpty = $("#historyEmpty");
const toast = $("#toast");
const inferenceProgress = $("#inferenceProgress");
const progressPercent = $("#progressPercent");
const progressDetail = $("#progressDetail");
const img2imgControls = $("#img2imgControls");
const initImageInput = $("#initImageInput");
const initImageName = $("#initImageName");
const maskEditor = $("#maskEditor");
const maskSourceCanvas = $("#maskSourceCanvas");
const maskPaintCanvas = $("#maskPaintCanvas");
const maskImageName = $("#maskImageName");
const ipAdapterImageInput = $("#ipAdapterImageInput");
const ipAdapterImageName = $("#ipAdapterImageName");
let maskTool = "brush";
let maskDrawing = false;
let maskDirty = false;
let lastMaskPoint = null;
let historyItems = [];
let toastTimer;

function notify(message, error = false) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className = `toast show${error ? " error" : ""}`;
  toastTimer = setTimeout(() => { toast.className = "toast"; }, 4500);
}

function updateIpAdapterControls() {
  const active = Boolean(ipAdapterSelect.value);
  $("#ipAdapterControls").hidden = !active;
  ipAdapterImageInput.required = active && !ipAdapterImageName.value;
}

function showIpAdapterImage(filename, imageUrl) {
  ipAdapterImageName.value = filename || "";
  $("#ipAdapterImageThumbnail").src = imageUrl || "";
  $("#ipAdapterImagePreview").hidden = !filename;
  updateIpAdapterControls();
}

async function uploadIpAdapterImage() {
  const file = ipAdapterImageInput.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  ipAdapterImageInput.disabled = true;
  try {
    const uploaded = await api("/api/images/upload", { method: "POST", body: formData });
    showIpAdapterImage(uploaded.filename, uploaded.image_url);
    notify("IP-Adapter 图像提示已上传到本机");
  } catch (error) {
    ipAdapterImageInput.value = "";
    notify(error.message, true);
  } finally { ipAdapterImageInput.disabled = false; }
}

async function api(path, options = {}) {
  const headers = options.body instanceof FormData
    ? { ...(options.headers || {}) }
    : { "Content-Type": "application/json", ...(options.headers || {}) };
  const response = await fetch(path, { ...options, headers });
  let body;
  try { body = await response.json(); } catch { body = null; }
  if (!response.ok) {
    const detail = body?.detail;
    const message = Array.isArray(detail) ? detail.map((item) => item.msg).join("；") : detail;
    throw new Error(message || `请求失败（HTTP ${response.status}）`);
  }
  return body;
}

function fillSelect(select, items, emptyLabel, selectedValue = "") {
  select.replaceChildren(new Option(emptyLabel, ""));
  items.forEach((item) => select.add(new Option(item, item)));
  if (items.includes(selectedValue)) select.value = selectedValue;
}

async function loadStatus() {
  const data = await api("/api/status");
  const device = $("#deviceBadge");
  device.querySelector("span").textContent = data.device.label;
  if (!data.device.cuda_available) {
    device.classList.add("muted");
    device.title = "未检测到可用 CUDA，将使用速度较慢的 CPU 生成";
    notify("未检测到 CUDA，当前使用 CPU；SDXL 生成速度会很慢。", true);
  }
  const sampler = $("#sampler");
  sampler.replaceChildren(...data.samplers.map((name) => new Option(name, name)));
  sampler.value = data.defaults.sampler;
}

async function refreshWeights(showMessage = true) {
  const oldModel = modelSelect.value;
  const oldLora = loraSelect.value;
  const oldIpAdapter = ipAdapterSelect.value;
  refreshButton.disabled = true;
  try {
    const data = await api("/api/weights/refresh", { method: "POST" });
    fillSelect(modelSelect, data.models, data.models.length ? "选择 SDXL 模型" : "models 目录中没有模型", oldModel);
    fillSelect(loraSelect, data.loras, "不使用 LoRA", oldLora);
    fillSelect(ipAdapterSelect, data.ip_adapters, "不使用 IP-Adapter", oldIpAdapter);
    if (!modelSelect.value && data.models.length === 1) modelSelect.value = data.models[0];
    updateModelState();
    if (showMessage) notify(`已扫描到 ${data.models.length} 个模型、${data.loras.length} 个 LoRA、${data.ip_adapters.length} 个 IP-Adapter`);
  } catch (error) { notify(error.message, true); }
  finally { refreshButton.disabled = false; }
}

function updateModelState() {
  const state = $("#modelState");
  state.querySelector("span").textContent = modelSelect.value || "尚未选择模型";
  state.classList.toggle("muted", !modelSelect.value);
}

function collectParameters() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  return {
    mode,
    model: modelSelect.value,
    lora: loraSelect.value || null,
    lora_scale: Number($("#loraScale").value),
    ip_adapter: ipAdapterSelect.value || null,
    ip_adapter_image: ipAdapterSelect.value ? ipAdapterImageName.value || null : null,
    ip_adapter_scale: Number($("#ipAdapterScale").value),
    prompt: $("#prompt").value.trim(),
    negative_prompt: $("#negativePrompt").value.trim(),
    width: Number($("#width").value),
    height: Number($("#height").value),
    cfg: Number($("#cfg").value),
    steps: Number($("#steps").value),
    sampler: $("#sampler").value,
    clip_skip: Number($("#clipSkip").value),
    seed: Number($("#seed").value),
    batch_size: Number($("#batchSize").value),
    init_image: mode === "img2img" ? initImageName.value || null : null,
    mask_image: mode === "img2img" ? maskImageName.value || null : null,
    strength: Number($("#strength").value),
  };
}

function setMode(mode) {
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  img2imgControls.hidden = mode !== "img2img";
  initImageInput.required = mode === "img2img" && !initImageName.value;
}

function showInitImage(filename, imageUrl, maskFilename = "", maskUrl = "") {
  initImageName.value = filename || "";
  $("#initImageThumbnail").src = imageUrl || "";
  $("#initImagePreview").hidden = !filename;
  maskEditor.hidden = !filename;
  initImageInput.required = !filename && !img2imgControls.hidden;
  if (filename && imageUrl) loadMaskSource(imageUrl, maskFilename, maskUrl);
  else resetMaskEditor();
}

async function uploadInitImage() {
  const file = initImageInput.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  initImageInput.disabled = true;
  try {
    const uploaded = await api("/api/images/upload", { method: "POST", body: formData });
    showInitImage(uploaded.filename, uploaded.image_url);
    notify("参考图已上传到本机");
  } catch (error) {
    initImageInput.value = "";
    notify(error.message, true);
  } finally { initImageInput.disabled = false; }
}

function resetMaskEditor() {
  maskImageName.value = "";
  maskDirty = false;
  lastMaskPoint = null;
  $("#maskState").textContent = "未绘制";
  maskPaintCanvas.getContext("2d").clearRect(0, 0, maskPaintCanvas.width, maskPaintCanvas.height);
}

function loadMaskSource(imageUrl, maskFilename = "", maskUrl = "") {
  const source = new Image();
  source.onload = () => {
    [maskSourceCanvas, maskPaintCanvas].forEach((canvas) => {
      canvas.width = source.naturalWidth;
      canvas.height = source.naturalHeight;
    });
    maskSourceCanvas.getContext("2d").drawImage(source, 0, 0);
    resetMaskEditor();
    if (maskUrl) loadExistingMask(maskUrl, maskFilename);
  };
  source.src = `${imageUrl}${imageUrl.includes("?") ? "&" : "?"}t=${Date.now()}`;
}

function loadExistingMask(maskUrl, maskFilename) {
  const image = new Image();
  image.onload = () => {
    const context = maskPaintCanvas.getContext("2d");
    const buffer = document.createElement("canvas");
    buffer.width = maskPaintCanvas.width;
    buffer.height = maskPaintCanvas.height;
    const bufferContext = buffer.getContext("2d");
    bufferContext.drawImage(image, 0, 0, buffer.width, buffer.height);
    const pixels = bufferContext.getImageData(0, 0, buffer.width, buffer.height);
    for (let index = 0; index < pixels.data.length; index += 4) {
      const maskValue = pixels.data[index];
      pixels.data[index] = 226;
      pixels.data[index + 1] = 109;
      pixels.data[index + 2] = 90;
      pixels.data[index + 3] = Math.round(maskValue * 0.48);
    }
    context.putImageData(pixels, 0, 0);
    maskImageName.value = maskFilename;
    maskDirty = true;
    $("#maskState").textContent = "已载入";
  };
  image.src = `${maskUrl}?t=${Date.now()}`;
}

function maskPoint(event) {
  const rect = maskPaintCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * maskPaintCanvas.width / rect.width,
    y: (event.clientY - rect.top) * maskPaintCanvas.height / rect.height,
  };
}

function drawMask(from, to) {
  const context = maskPaintCanvas.getContext("2d");
  context.save();
  context.globalCompositeOperation = maskTool === "eraser" ? "destination-out" : "source-over";
  context.strokeStyle = "rgba(226,109,90,.48)";
  context.lineWidth = Number($("#maskBrushSize").value) * maskPaintCanvas.width / Math.max(maskPaintCanvas.clientWidth, 1);
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(from.x, from.y);
  context.lineTo(to.x, to.y);
  context.stroke();
  context.restore();
  maskDirty = true;
  maskImageName.value = "";
  $("#maskState").textContent = "待上传";
}

function setMaskTool(tool) {
  maskTool = tool;
  ["maskBrush", "maskEraser"].forEach((id) => {
    const active = id === (tool === "brush" ? "maskBrush" : "maskEraser");
    $(`#${id}`).classList.toggle("active", active);
    $(`#${id}`).setAttribute("aria-pressed", String(active));
  });
}

async function uploadMask() {
  if (!maskDirty || maskImageName.value) return;
  const output = document.createElement("canvas");
  output.width = maskPaintCanvas.width;
  output.height = maskPaintCanvas.height;
  const outputContext = output.getContext("2d");
  const sourcePixels = maskPaintCanvas.getContext("2d").getImageData(0, 0, output.width, output.height);
  const pixels = outputContext.createImageData(output.width, output.height);
  for (let index = 0; index < pixels.data.length; index += 4) {
    const value = sourcePixels.data[index + 3] ? 255 : 0;
    pixels.data[index] = value;
    pixels.data[index + 1] = value;
    pixels.data[index + 2] = value;
    pixels.data[index + 3] = 255;
  }
  outputContext.putImageData(pixels, 0, 0);
  const blob = await new Promise((resolve) => output.toBlob(resolve, "image/png"));
  const formData = new FormData();
  formData.append("file", blob, "mask.png");
  const uploaded = await api("/api/images/upload", { method: "POST", body: formData });
  maskImageName.value = uploaded.filename;
  $("#maskState").textContent = "已上传";
}

function setGenerating(active) {
  generateButton.disabled = active;
  generateButton.querySelector(".button-label").textContent = active ? "正在生成…" : "开始生成";
  loadingState.hidden = !active;
  if (active) { emptyState.hidden = true; previewGrid.replaceChildren(); }
}

function updateProgress(task) {
  const value = Math.max(0, Math.min(100, task.progress ?? 0));
  inferenceProgress.value = value;
  inferenceProgress.textContent = `${value}%`;
  progressPercent.value = `${value}%`;
  progressDetail.textContent = task.status === "queued"
    ? task.message
    : `${task.message} · ${task.current_step}/${task.total_steps} steps`;
}

function wait(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }

async function waitForTask(taskId) {
  while (true) {
    const task = await api(`/api/tasks/${taskId}`);
    updateProgress(task);
    if (task.status === "completed") return task;
    if (task.status === "failed") throw new Error(task.error || "生成失败，请重试。");
    await wait(250);
  }
}

async function generate(event) {
  event.preventDefault();
  if (!form.reportValidity()) return;
  let parameters = collectParameters();
  if (!parameters.model) return notify("请先刷新并选择一个 SDXL 模型。", true);
  if (parameters.mode === "img2img" && !parameters.init_image) return notify("请先上传图生图参考图。", true);
  if (parameters.ip_adapter && !parameters.ip_adapter_image) return notify("请上传 IP-Adapter 图像提示。", true);
  setGenerating(true);
  updateProgress({ progress: 0, status: "queued", message: "任务已进入队列" });
  try {
    if (parameters.mode === "img2img" && maskDirty) {
      updateProgress({ progress: 0, status: "queued", message: "正在上传局部重绘蒙版" });
      await uploadMask();
      parameters = collectParameters();
    }
    const accepted = await api("/api/generate", { method: "POST", body: JSON.stringify(parameters) });
    const result = await waitForTask(accepted.task_id);
    previewGrid.replaceChildren(...result.images.map((item) => {
      const image = new Image();
      image.src = `${item.image_url}?t=${Date.now()}`;
      image.alt = `生成结果，种子 ${item.seed}`;
      return image;
    }));
    notify(`已完成 ${result.images.length} 张图片`);
    await loadHistory();
  } catch (error) {
    emptyState.hidden = false;
    notify(error.message, true);
  } finally { setGenerating(false); }
}

function historyCard(item) {
  const article = document.createElement("article");
  article.className = "history-card";
  const image = new Image();
  image.src = item.image_url;
  image.alt = item.prompt ? `历史生成图：${item.prompt.slice(0, 80)}` : "历史生成图";
  image.loading = "lazy";
  const apply = document.createElement("button");
  apply.className = "apply-button";
  apply.type = "button";
  apply.textContent = "应用参数";
  apply.addEventListener("click", () => applyParameters(item));
  const remove = document.createElement("button");
  remove.className = "delete-button";
  remove.type = "button";
  remove.textContent = "删除";
  remove.setAttribute("aria-label", `删除历史图片：${(item.prompt || "无提示词").slice(0, 40)}`);
  remove.addEventListener("click", async () => {
    if (!window.confirm("确定删除这张历史图片及其参数记录吗？此操作无法撤销。")) return;
    remove.disabled = true;
    try {
      await api(`/api/history/${item.id}`, { method: "DELETE" });
      notify("历史图片已删除");
      await loadHistory();
    } catch (error) {
      remove.disabled = false;
      notify(error.message, true);
    }
  });
  const info = document.createElement("div");
  info.className = "history-info";
  const prompt = document.createElement("p");
  prompt.className = "history-prompt";
  prompt.textContent = item.prompt || "（无提示词）";
  const meta = document.createElement("div");
  meta.className = "history-meta";
  const modeLabel = item.mask_image
    ? `局部重绘 ${item.strength}`
    : item.mode === "img2img" ? `图生图 ${item.strength}` : "文生图";
  meta.textContent = `${modeLabel} · ${item.width}×${item.height} · ${item.sampler} · CLIP ${item.clip_skip ?? 2} · ${item.steps} steps · seed ${item.seed}`;
  info.append(prompt, meta);
  article.append(image, apply, remove, info);
  return article;
}

async function loadHistory() {
  historyItems = await api("/api/history");
  historyGrid.replaceChildren(...historyItems.map(historyCard));
  historyEmpty.hidden = historyItems.length > 0;
  $("#historyCount").textContent = `${historyItems.length} IMAGE${historyItems.length === 1 ? "" : "S"}`;
}

async function ensureOption(select, value, fallbackLabel) {
  if (!value) { select.value = ""; return; }
  if (![...select.options].some((option) => option.value === value)) {
    select.add(new Option(`${value}（当前未扫描到）`, value));
  }
  select.value = value || fallbackLabel;
}

function applyParameters(item) {
  setMode(item.mode ?? "txt2img");
  ensureOption(modelSelect, item.model, "");
  ensureOption(loraSelect, item.lora, "");
  ensureOption(ipAdapterSelect, item.ip_adapter, "");
  $("#loraScale").value = item.lora_scale ?? 1;
  $("#loraScaleValue").value = Number(item.lora_scale ?? 1).toFixed(2);
  $("#ipAdapterScale").value = item.ip_adapter_scale ?? 0.6;
  $("#ipAdapterScaleValue").value = Number(item.ip_adapter_scale ?? 0.6).toFixed(2);
  showIpAdapterImage(item.ip_adapter_image ?? "", item.ip_adapter_image_url ?? "");
  $("#prompt").value = item.prompt ?? "";
  $("#negativePrompt").value = item.negative_prompt ?? "";
  $("#width").value = item.width ?? 832;
  $("#height").value = item.height ?? 1216;
  $("#cfg").value = item.cfg ?? 4;
  $("#steps").value = item.steps ?? 30;
  ensureOption($("#sampler"), item.sampler ?? "Euler a", "Euler a");
  $("#clipSkip").value = item.clip_skip ?? 2;
  $("#seed").value = item.seed ?? -1;
  $("#batchSize").value = item.batch_size ?? 1;
  $("#strength").value = item.strength ?? 0.65;
  $("#strengthValue").value = Number(item.strength ?? 0.65).toFixed(2);
  showInitImage(
    item.init_image ?? "",
    item.init_image_url ?? "",
    item.mask_image ?? "",
    item.mask_image_url ?? "",
  );
  updateModelState();
  updateStageMeta();
  window.scrollTo({ top: 0, behavior: "smooth" });
  notify("已应用该图片的全部生成参数");
}

function updateStageMeta() { $("#stageMeta").textContent = `${$("#width").value} × ${$("#height").value}`; }
refreshButton.addEventListener("click", () => refreshWeights(true));
modelSelect.addEventListener("change", updateModelState);
ipAdapterSelect.addEventListener("change", updateIpAdapterControls);
form.addEventListener("submit", generate);
$("#loraScale").addEventListener("input", (event) => { $("#loraScaleValue").value = Number(event.target.value).toFixed(2); });
$("#ipAdapterScale").addEventListener("input", (event) => { $("#ipAdapterScaleValue").value = Number(event.target.value).toFixed(2); });
ipAdapterImageInput.addEventListener("change", uploadIpAdapterImage);
$("#removeIpAdapterImage").addEventListener("click", () => { ipAdapterImageInput.value = ""; showIpAdapterImage("", ""); });
document.querySelectorAll('input[name="mode"]').forEach((input) => input.addEventListener("change", () => setMode(input.value)));
initImageInput.addEventListener("change", uploadInitImage);
$("#removeInitImage").addEventListener("click", () => { initImageInput.value = ""; showInitImage("", ""); });
$("#maskBrush").addEventListener("click", () => setMaskTool("brush"));
$("#maskEraser").addEventListener("click", () => setMaskTool("eraser"));
$("#clearMask").addEventListener("click", resetMaskEditor);
$("#maskBrushSize").addEventListener("input", (event) => { $("#maskBrushSizeValue").value = event.target.value; });
maskPaintCanvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  maskPaintCanvas.setPointerCapture(event.pointerId);
  maskDrawing = true;
  lastMaskPoint = maskPoint(event);
  drawMask(lastMaskPoint, lastMaskPoint);
});
maskPaintCanvas.addEventListener("pointermove", (event) => {
  if (!maskDrawing) return;
  const point = maskPoint(event);
  drawMask(lastMaskPoint, point);
  lastMaskPoint = point;
});
maskPaintCanvas.addEventListener("pointerup", () => { maskDrawing = false; lastMaskPoint = null; });
maskPaintCanvas.addEventListener("pointercancel", () => { maskDrawing = false; lastMaskPoint = null; });
$("#strength").addEventListener("input", (event) => { $("#strengthValue").value = Number(event.target.value).toFixed(2); });
$("#width").addEventListener("input", updateStageMeta);
$("#height").addEventListener("input", updateStageMeta);
document.addEventListener("keydown", (event) => { if (event.ctrlKey && event.key === "Enter" && !generateButton.disabled) form.requestSubmit(); });

Promise.all([loadStatus(), refreshWeights(false), loadHistory()]).catch((error) => notify(error.message, true));
