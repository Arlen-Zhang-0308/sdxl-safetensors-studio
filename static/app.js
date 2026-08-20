const $ = (selector) => document.querySelector(selector);
const form = $("#generateForm");
const modelSelect = $("#model");
const loraSelect = $("#lora");
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
let historyItems = [];
let toastTimer;

function notify(message, error = false) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className = `toast show${error ? " error" : ""}`;
  toastTimer = setTimeout(() => { toast.className = "toast"; }, 4500);
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
  refreshButton.disabled = true;
  try {
    const data = await api("/api/weights/refresh", { method: "POST" });
    fillSelect(modelSelect, data.models, data.models.length ? "选择 SDXL 模型" : "models 目录中没有模型", oldModel);
    fillSelect(loraSelect, data.loras, "不使用 LoRA", oldLora);
    if (!modelSelect.value && data.models.length === 1) modelSelect.value = data.models[0];
    updateModelState();
    if (showMessage) notify(`已扫描到 ${data.models.length} 个模型、${data.loras.length} 个 LoRA`);
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
    strength: Number($("#strength").value),
  };
}

function setMode(mode) {
  document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
  img2imgControls.hidden = mode !== "img2img";
  initImageInput.required = mode === "img2img" && !initImageName.value;
}

function showInitImage(filename, imageUrl) {
  initImageName.value = filename || "";
  $("#initImageThumbnail").src = imageUrl || "";
  $("#initImagePreview").hidden = !filename;
  initImageInput.required = !filename && !img2imgControls.hidden;
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
  const parameters = collectParameters();
  if (!parameters.model) return notify("请先刷新并选择一个 SDXL 模型。", true);
  if (parameters.mode === "img2img" && !parameters.init_image) return notify("请先上传图生图参考图。", true);
  setGenerating(true);
  updateProgress({ progress: 0, status: "queued", message: "任务已进入队列" });
  try {
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
  const modeLabel = item.mode === "img2img" ? `图生图 ${item.strength}` : "文生图";
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
  $("#loraScale").value = item.lora_scale ?? 1;
  $("#loraScaleValue").value = Number(item.lora_scale ?? 1).toFixed(2);
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
  showInitImage(item.init_image ?? "", item.init_image_url ?? "");
  updateModelState();
  updateStageMeta();
  window.scrollTo({ top: 0, behavior: "smooth" });
  notify("已应用该图片的全部生成参数");
}

function updateStageMeta() { $("#stageMeta").textContent = `${$("#width").value} × ${$("#height").value}`; }
refreshButton.addEventListener("click", () => refreshWeights(true));
modelSelect.addEventListener("change", updateModelState);
form.addEventListener("submit", generate);
$("#loraScale").addEventListener("input", (event) => { $("#loraScaleValue").value = Number(event.target.value).toFixed(2); });
document.querySelectorAll('input[name="mode"]').forEach((input) => input.addEventListener("change", () => setMode(input.value)));
initImageInput.addEventListener("change", uploadInitImage);
$("#removeInitImage").addEventListener("click", () => { initImageInput.value = ""; showInitImage("", ""); });
$("#strength").addEventListener("input", (event) => { $("#strengthValue").value = Number(event.target.value).toFixed(2); });
$("#width").addEventListener("input", updateStageMeta);
$("#height").addEventListener("input", updateStageMeta);
document.addEventListener("keydown", (event) => { if (event.ctrlKey && event.key === "Enter" && !generateButton.disabled) form.requestSubmit(); });

Promise.all([loadStatus(), refreshWeights(false), loadHistory()]).catch((error) => notify(error.message, true));
