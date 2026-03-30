const tabSingle = document.getElementById("tabSingle");
const tabBatch = document.getElementById("tabBatch");
const panelSingle = document.getElementById("panelSingle");
const panelBatch = document.getElementById("panelBatch");

const modelSelect = document.getElementById("modelSelect");
const tileInput = document.getElementById("tileInput");
const deviceBadge = document.getElementById("deviceBadge");

const singleDropZone = document.getElementById("singleDropZone");
const singlePickBtn = document.getElementById("singlePickBtn");
const singleImageInput = document.getElementById("singleImageInput");
const singleForm = document.getElementById("singleForm");
const singleRunBtn = document.getElementById("singleRunBtn");
const singleStatus = document.getElementById("singleStatus");
const singleOutputPath = document.getElementById("singleOutputPath");
const singleFileMeta = document.getElementById("singleFileMeta");
const singleDoneMetric = document.getElementById("singleDoneMetric");
const batchDoneMetric = document.getElementById("batchDoneMetric");
const singleRunInfo = document.getElementById("singleRunInfo");
const singleSizeInfo = document.getElementById("singleSizeInfo");

const beforePreview = document.getElementById("beforePreview");
const afterPreview = document.getElementById("afterPreview");
const beforeHint = document.getElementById("beforeHint");
const afterHint = document.getElementById("afterHint");

const openResultBtn = document.getElementById("openResultBtn");
const copyResultPathBtn = document.getElementById("copyResultPathBtn");

const folderPickBtn = document.getElementById("folderPickBtn");
const folderInput = document.getElementById("folderInput");
const folderInfo = document.getElementById("folderInfo");
const batchForm = document.getElementById("batchForm");
const batchRunBtn = document.getElementById("batchRunBtn");
const batchStatus = document.getElementById("batchStatus");
const batchSummary = document.getElementById("batchSummary");
const batchOutputPath = document.getElementById("batchOutputPath");
const batchPreviewGrid = document.getElementById("batchPreviewGrid");
const batchErrorList = document.getElementById("batchErrorList");
const batchRunInfo = document.getElementById("batchRunInfo");
const batchTimeInfo = document.getElementById("batchTimeInfo");

const activityLog = document.getElementById("activityLog");
const clearLogBtn = document.getElementById("clearLogBtn");

let lastSingleResultUrl = "";
let lastSingleSavedPath = "";

const addLog = (message, level = "info") => {
  if (!activityLog) return;
  const item = document.createElement("li");
  const t = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  item.textContent = `[${t}] ${message}`;
  if (level === "error") item.style.color = "#c03232";
  if (level === "success") item.style.color = "#1f6fb6";
  activityLog.prepend(item);
  while (activityLog.childElementCount > 40) {
    activityLog.removeChild(activityLog.lastChild);
  }
};

const setStatus = (el, text, state = "normal") => {
  if (!el) return;
  el.textContent = text;
  if (state === "error") el.style.color = "#c03232";
  else if (state === "ok") el.style.color = "#1f6fb6";
  else el.style.color = "";
};

const setActiveTab = (mode) => {
  const isSingle = mode === "single";
  tabSingle.classList.toggle("is-active", isSingle);
  tabBatch.classList.toggle("is-active", !isSingle);
  panelSingle.classList.toggle("is-active", isSingle);
  panelBatch.classList.toggle("is-active", !isSingle);
  addLog(`切换到${isSingle ? "单图演示" : "批量处理"}模式`);
};

const resetSingleResultState = () => {
  lastSingleResultUrl = "";
  lastSingleSavedPath = "";
  if (openResultBtn) openResultBtn.disabled = true;
  if (copyResultPathBtn) copyResultPathBtn.disabled = true;
  afterPreview.removeAttribute("src");
  if (afterHint) {
    afterHint.textContent = "处理完成后显示结果图";
    afterHint.style.display = "block";
  }
};

const setSinglePreviewFromFile = (file) => {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    beforePreview.src = ev.target?.result;
    if (beforeHint) beforeHint.style.display = "none";
    if (singleFileMeta) {
      singleFileMeta.textContent = `文件大小：${(file.size / 1024 / 1024).toFixed(2)} MB`;
    }
    resetSingleResultState();
    setStatus(singleStatus, `已选择：${file.name}`);
    setStatus(singleRunInfo, "本次运行：待开始");
    setStatus(singleSizeInfo, "输入/输出尺寸：-");
    addLog(`已选择单图文件 ${file.name}`);
  };
  reader.readAsDataURL(file);
};

tabSingle.addEventListener("click", () => setActiveTab("single"));
tabBatch.addEventListener("click", () => setActiveTab("batch"));

singlePickBtn.addEventListener("click", () => singleImageInput.click());
singleImageInput.addEventListener("change", () => {
  const file = singleImageInput.files?.[0];
  setSinglePreviewFromFile(file);
});

singleDropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  singleDropZone.classList.add("is-dragover");
});
singleDropZone.addEventListener("dragleave", () => {
  singleDropZone.classList.remove("is-dragover");
});
singleDropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  singleDropZone.classList.remove("is-dragover");
  if (!e.dataTransfer?.files?.length) return;
  const [file] = e.dataTransfer.files;
  const dt = new DataTransfer();
  dt.items.add(file);
  singleImageInput.files = dt.files;
  setSinglePreviewFromFile(file);
});
singleDropZone.addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") singleImageInput.click();
});

singleForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = singleImageInput.files?.[0];
  if (!file) {
    setStatus(singleStatus, "请先选择一张图片。", "error");
    addLog("单图处理失败：未选择图片", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", file, file.name);
  formData.append("model_key", modelSelect.value);
  formData.append("tile", tileInput.value || "256");

  singleRunBtn.disabled = true;
  setStatus(singleStatus, "正在处理，请稍候...");
  setStatus(singleRunInfo, `本次运行：模型 ${modelSelect.value}，Tile ${tileInput.value || "256"}`);
  setStatus(singleSizeInfo, "输入/输出尺寸：计算中...");
  singleOutputPath.textContent = "";
  if (afterHint) {
    afterHint.textContent = "处理中...";
    afterHint.style.display = "block";
  }
  afterPreview.removeAttribute("src");
  addLog("开始执行单图推理");

  try {
    const resp = await fetch("/api/process-single", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || "处理失败");
    }

    const bust = `${data.output_url}?v=${Date.now()}`;
    afterPreview.src = bust;
    if (afterHint) afterHint.style.display = "none";

    const timeText = typeof data.elapsed_ms === "number" ? `，耗时 ${data.elapsed_ms} ms` : "";
    const devText = data.device ? `（${data.device}）` : "";
    setStatus(singleStatus, `完成：${data.model}${devText}${timeText}`, "ok");

    const inSize = data.input_size
      ? `${data.input_size.width}x${data.input_size.height}`
      : "-";
    const outSize = data.output_size
      ? `${data.output_size.width}x${data.output_size.height}`
      : "-";
    setStatus(singleRunInfo, `本次运行：run_id=${data.run_id || "-"}`);
    setStatus(singleSizeInfo, `输入/输出尺寸：${inSize} -> ${outSize}`);

    singleOutputPath.textContent = `输出：${data.saved_to}`;
    lastSingleResultUrl = data.output_url;
    lastSingleSavedPath = data.saved_to || "";
    if (openResultBtn) openResultBtn.disabled = false;
    if (copyResultPathBtn) copyResultPathBtn.disabled = false;

    if (singleDoneMetric) {
      singleDoneMetric.textContent = String(Number(singleDoneMetric.textContent || "0") + 1);
    }
    addLog(`单图推理完成，run_id=${data.run_id || "-"}`, "success");
  } catch (err) {
    if (afterHint) {
      afterHint.textContent = "处理失败，请重试";
      afterHint.style.display = "block";
    }
    setStatus(singleStatus, `失败：${err.message}`, "error");
    setStatus(singleRunInfo, "本次运行：失败");
    setStatus(singleSizeInfo, "输入/输出尺寸：-");
    addLog(`单图推理失败：${err.message}`, "error");
  } finally {
    singleRunBtn.disabled = false;
  }
});

openResultBtn.addEventListener("click", () => {
  if (!lastSingleResultUrl) return;
  window.open(lastSingleResultUrl, "_blank", "noopener,noreferrer");
  addLog("已打开结果图");
});

copyResultPathBtn.addEventListener("click", async () => {
  if (!lastSingleSavedPath) return;
  try {
    await navigator.clipboard.writeText(lastSingleSavedPath);
    setStatus(singleStatus, "已复制结果路径", "ok");
    addLog("已复制结果路径");
  } catch {
    setStatus(singleStatus, "复制失败，请手动复制路径", "error");
    addLog("复制路径失败", "error");
  }
});

folderPickBtn.addEventListener("click", () => folderInput.click());
folderInput.addEventListener("change", () => {
  const files = Array.from(folderInput.files || []);
  if (!files.length) {
    setStatus(folderInfo, "尚未选择文件夹");
    return;
  }
  const root =
    files[0].webkitRelativePath && files[0].webkitRelativePath.includes("/")
      ? files[0].webkitRelativePath.split("/")[0]
      : "(无目录名)";
  setStatus(folderInfo, `已选择 ${files.length} 个文件，目录：${root}`);
  addLog(`已选择批量目录 ${root}，共 ${files.length} 个文件`);
});

batchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = Array.from(folderInput.files || []);
  if (!files.length) {
    setStatus(batchStatus, "请先选择文件夹。", "error");
    addLog("批量处理失败：未选择文件夹", "error");
    return;
  }

  batchRunBtn.disabled = true;
  setStatus(batchStatus, "上传并处理中，请稍候...");
  setStatus(batchRunInfo, "本次运行：处理中...");
  setStatus(batchTimeInfo, "总耗时：计算中...");
  batchSummary.textContent = "";
  batchOutputPath.textContent = "";
  batchPreviewGrid.innerHTML = "";
  batchErrorList.innerHTML = "";
  addLog(`开始批量处理，共 ${files.length} 个上传项`);

  const formData = new FormData();
  formData.append("model_key", modelSelect.value);
  formData.append("tile", tileInput.value || "256");
  files.forEach((file) => formData.append("files", file, file.webkitRelativePath || file.name));

  try {
    const resp = await fetch("/api/process-batch", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || "批量处理失败");
    }

    setStatus(batchStatus, `完成：${data.model}`, "ok");
    batchSummary.textContent = `处理成功 ${data.processed} 张，跳过 ${data.skipped} 张。`;
    batchOutputPath.textContent = `输出目录：${data.output_dir}`;
    setStatus(batchRunInfo, `本次运行：run_id=${data.run_id || "-"}`);
    setStatus(batchTimeInfo, `总耗时：${typeof data.elapsed_ms === "number" ? `${data.elapsed_ms} ms` : "-"}`);

    if (batchDoneMetric) {
      batchDoneMetric.textContent = String(
        Number(batchDoneMetric.textContent || "0") + Number(data.processed || 0)
      );
    }

    if (Array.isArray(data.preview_images)) {
      data.preview_images.forEach((item) => {
        const a = document.createElement("a");
        a.href = item.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        const img = document.createElement("img");
        img.src = item.url;
        img.alt = item.name;
        a.appendChild(img);
        batchPreviewGrid.appendChild(a);
      });
    }

    if (Array.isArray(data.errors) && data.errors.length) {
      data.errors.forEach((err) => {
        const li = document.createElement("li");
        li.textContent = err;
        batchErrorList.appendChild(li);
      });
    }

    addLog(
      `批量处理完成：成功 ${data.processed}，跳过 ${data.skipped}，耗时 ${data.elapsed_ms ?? "-"} ms`,
      "success"
    );
  } catch (err) {
    setStatus(batchStatus, `失败：${err.message}`, "error");
    setStatus(batchRunInfo, "本次运行：失败");
    setStatus(batchTimeInfo, "总耗时：-");
    addLog(`批量处理失败：${err.message}`, "error");
  } finally {
    batchRunBtn.disabled = false;
  }
});

clearLogBtn.addEventListener("click", () => {
  activityLog.innerHTML = "";
  addLog("日志已清空");
});

const bootstrap = async () => {
  try {
    const resp = await fetch("/api/models");
    const data = await resp.json();
    if (data.ok) {
      deviceBadge.textContent = `推理设备：${data.device}`;
      addLog(`初始化完成，当前设备 ${data.device}`, "success");
    } else {
      deviceBadge.textContent = "推理设备：未知";
      addLog("初始化失败：设备未知", "error");
    }
  } catch {
    deviceBadge.textContent = "推理设备：接口不可用";
    addLog("初始化失败：接口不可用", "error");
  }
};

bootstrap();
