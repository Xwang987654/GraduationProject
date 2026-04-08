const tabSingle = document.getElementById("tabSingle");
const tabBatch = document.getElementById("tabBatch");
const panelSingle = document.getElementById("panelSingle");
const panelBatch = document.getElementById("panelBatch");

const modelSelect = document.getElementById("modelSelect");
const tileInput = document.getElementById("tileInput");
const deviceBadge = document.getElementById("deviceBadge");
const singleWorkflowSelect = document.getElementById("singleWorkflowSelect");
const singleWorkflowHint = document.getElementById("singleWorkflowHint");

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
const beforeHead = document.getElementById("beforeHead");
const afterHead = document.getElementById("afterHead");
const lqCard = document.getElementById("lqCard");
const lqPreview = document.getElementById("lqPreview");
const lqHint = document.getElementById("lqHint");

const openResultBtn = document.getElementById("openResultBtn");
const copyResultPathBtn = document.getElementById("copyResultPathBtn");

const folderPickBtn = document.getElementById("folderPickBtn");
const folderInput = document.getElementById("folderInput");
const folderInfo = document.getElementById("folderInfo");
const batchCompatHint = document.getElementById("batchCompatHint");
const batchForm = document.getElementById("batchForm");
const batchRunBtn = document.getElementById("batchRunBtn");
const batchCancelBtn = document.getElementById("batchCancelBtn");
const batchStatus = document.getElementById("batchStatus");
const batchSummary = document.getElementById("batchSummary");
const batchOutputPath = document.getElementById("batchOutputPath");
const batchPreviewGrid = document.getElementById("batchPreviewGrid");
const batchErrorList = document.getElementById("batchErrorList");
const batchRunInfo = document.getElementById("batchRunInfo");
const batchTimeInfo = document.getElementById("batchTimeInfo");
const batchProgressBar = document.getElementById("batchProgressBar");
const batchProgressText = document.getElementById("batchProgressText");

const activityLog = document.getElementById("activityLog");
const clearLogBtn = document.getElementById("clearLogBtn");

const dbLogDetails = document.getElementById("dbLogDetails");
const logRunIdInput = document.getElementById("logRunIdInput");
const logModeSelect = document.getElementById("logModeSelect");
const logStatusSelect = document.getElementById("logStatusSelect");
const logFromInput = document.getElementById("logFromInput");
const logToInput = document.getElementById("logToInput");
const logLimitSelect = document.getElementById("logLimitSelect");
const queryLogsBtn = document.getElementById("queryLogsBtn");
const latestLogsBtn = document.getElementById("latestLogsBtn");
const clearDbViewBtn = document.getElementById("clearDbViewBtn");
const logsPrevBtn = document.getElementById("logsPrevBtn");
const logsNextBtn = document.getElementById("logsNextBtn");
const logsPageInfo = document.getElementById("logsPageInfo");
const dbLogStatus = document.getElementById("dbLogStatus");
const dbLogList = document.getElementById("dbLogList");

let lastSingleResultUrl = "";
let lastSingleSavedPath = "";
let currentBatchRunId = "";
let batchPollTimer = null;
let currentBatchStartedAt = 0;
let metricUpdatedForBatchRun = "";
let lqGeneratorReady = false;

const DIRECT_WORKFLOW = "direct_lq_to_sr";
const REAL_TO_LQ_WORKFLOW = "real_hq_to_lq_to_sr";

const logQueryState = {
  runId: "",
  mode: "",
  status: "",
  createdFrom: "",
  createdTo: "",
  limit: 50,
  offset: 0,
  total: 0,
};

const addLog = (message, level = "info") => {
  if (!activityLog) return;
  const item = document.createElement("li");
  const t = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  item.textContent = `[${t}] ${message}`;
  if (level === "error") item.style.color = "#c03232";
  if (level === "success") item.style.color = "#1f6fb6";
  activityLog.prepend(item);
  while (activityLog.childElementCount > 80) {
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

const parseTile = () => {
  const valueRaw = String(tileInput.value || "256").trim();
  const value = Number(valueRaw);
  if (!Number.isInteger(value)) {
    throw new Error("Tile 必须是整数。");
  }
  if (value < 0 || value > 1024) {
    throw new Error("Tile 需在 0 到 1024 范围内。");
  }
  if (value !== 0 && value < 32) {
    throw new Error("Tile 为非 0 时，建议不小于 32。");
  }
  return String(value);
};

const getSingleWorkflowMode = () => {
  const mode = singleWorkflowSelect?.value || DIRECT_WORKFLOW;
  if (mode !== DIRECT_WORKFLOW && mode !== REAL_TO_LQ_WORKFLOW) {
    return DIRECT_WORKFLOW;
  }
  return mode;
};

const resetLqPreviewState = () => {
  if (lqPreview) lqPreview.removeAttribute("src");
  if (lqHint) {
    lqHint.textContent = "处理完成后显示生成低质量图";
    lqHint.style.display = "block";
  }
};

const applySingleWorkflowUI = (mode) => {
  const isRealWorkflow = mode === REAL_TO_LQ_WORKFLOW;
  if (lqCard) lqCard.hidden = !isRealWorkflow;
  if (beforeHead) beforeHead.textContent = isRealWorkflow ? "输入原图（现实照片）" : "输入低质量图";
  if (afterHead) afterHead.textContent = isRealWorkflow ? "SwinIR 重建高质量图" : "SwinIR 超分结果图";

  if (beforeHint && !beforePreview?.getAttribute("src")) {
    beforeHint.textContent = isRealWorkflow ? "请先上传现实照片" : "请先上传低质量图片";
    beforeHint.style.display = "block";
  }

  if (singleWorkflowHint) {
    if (isRealWorkflow) {
      singleWorkflowHint.textContent = "当前：输入现实照片，先生成低质量图，再执行 SwinIR 超分。";
    } else {
      singleWorkflowHint.textContent = "当前：直接输入低质量图，输出超分结果图。";
    }
    if (!lqGeneratorReady && isRealWorkflow) {
      singleWorkflowHint.textContent += "（当前环境未检测到可用 lq_generator）";
    }
  }

  if (!isRealWorkflow) {
    resetLqPreviewState();
  }
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
  openResultBtn.disabled = true;
  copyResultPathBtn.disabled = true;
  afterPreview.removeAttribute("src");
  resetLqPreviewState();
  if (afterHint) {
    afterHint.textContent = "处理完成后显示结果图";
    afterHint.style.display = "block";
  }
};

const setSinglePreviewFromFile = (file) => {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    const workflowMode = getSingleWorkflowMode();
    applySingleWorkflowUI(workflowMode);
    beforePreview.src = ev.target?.result;
    if (beforeHint) beforeHint.style.display = "none";
    singleFileMeta.textContent = `文件大小：${(file.size / 1024 / 1024).toFixed(2)} MB`;
    resetSingleResultState();
    setStatus(singleStatus, `已选择：${file.name}`);
    setStatus(singleRunInfo, `本次运行：待开始（${workflowMode}）`);
    setStatus(singleSizeInfo, "输入/输出尺寸：-");
    addLog(`已选择单图文件 ${file.name}`);
  };
  reader.readAsDataURL(file);
};

const resetBatchResult = () => {
  batchSummary.textContent = "";
  batchOutputPath.textContent = "";
  batchPreviewGrid.innerHTML = "";
  batchErrorList.innerHTML = "";
};

const setBatchProgress = (processed, skipped, failed, total) => {
  const done = Math.max(0, processed + skipped + failed);
  const safeTotal = Math.max(1, total || 0);
  const percent = total > 0 ? Math.round((done / total) * 100) : 0;
  if (batchProgressBar) {
    batchProgressBar.max = safeTotal;
    batchProgressBar.value = Math.min(done, safeTotal);
  }
  if (batchProgressText) {
    batchProgressText.textContent = `进度：${done}/${total || 0}（${percent}%）`;
  }
};

const stopBatchPolling = () => {
  if (batchPollTimer) {
    clearInterval(batchPollTimer);
    batchPollTimer = null;
  }
};

const formatDateTime = (raw) => {
  if (!raw) return "-";
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleString("zh-CN", { hour12: false });
};

const copyText = async (text, successMsg) => {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    addLog(successMsg, "success");
  } catch {
    addLog("复制失败，请手动复制", "error");
  }
};

tabSingle.addEventListener("click", () => setActiveTab("single"));
tabBatch.addEventListener("click", () => setActiveTab("batch"));

singleWorkflowSelect?.addEventListener("change", () => {
  const workflowMode = getSingleWorkflowMode();
  applySingleWorkflowUI(workflowMode);
  resetSingleResultState();
  singleOutputPath.textContent = "";
  setStatus(singleSizeInfo, workflowMode === REAL_TO_LQ_WORKFLOW ? "输入/低质/输出尺寸：-" : "输入/输出尺寸：-");
  addLog(`单图流程已切换：${workflowMode}`);
});

singlePickBtn.addEventListener("click", () => singleImageInput.click());
singleImageInput.addEventListener("change", () => {
  setSinglePreviewFromFile(singleImageInput.files?.[0]);
});

singleDropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  singleDropZone.classList.add("is-dragover");
});
singleDropZone.addEventListener("dragleave", () => singleDropZone.classList.remove("is-dragover"));
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

  let tile;
  try {
    tile = parseTile();
  } catch (err) {
    setStatus(singleStatus, err.message, "error");
    addLog(`单图处理失败：${err.message}`, "error");
    return;
  }

  const workflowMode = getSingleWorkflowMode();
  if (workflowMode === REAL_TO_LQ_WORKFLOW && !lqGeneratorReady) {
    setStatus(singleStatus, "当前环境 lq_generator 不可用，无法执行该流程。", "error");
    addLog("单图处理失败：lq_generator 未就绪", "error");
    return;
  }

  const formData = new FormData();
  formData.append("image", file, file.name);
  formData.append("model_key", modelSelect.value);
  formData.append("tile", tile);
  formData.append("workflow_mode", workflowMode);

  singleRunBtn.disabled = true;
  applySingleWorkflowUI(workflowMode);
  setStatus(singleStatus, "正在处理，请稍候...");
  setStatus(singleRunInfo, `本次运行：模型 ${modelSelect.value}，Tile ${tile}，流程 ${workflowMode}`);
  setStatus(singleSizeInfo, workflowMode === REAL_TO_LQ_WORKFLOW ? "输入/低质/输出尺寸：计算中..." : "输入/输出尺寸：计算中...");
  singleOutputPath.textContent = "";
  resetSingleResultState();
  if (afterHint) {
    afterHint.textContent = "处理中...";
    afterHint.style.display = "block";
  }
  if (workflowMode === REAL_TO_LQ_WORKFLOW && lqHint) {
    lqHint.textContent = "正在生成低质量图...";
    lqHint.style.display = "block";
  }
  addLog(`开始执行单图推理（${workflowMode}）`);

  try {
    const resp = await fetch("/api/process-single", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || "处理失败");
    }

    afterPreview.src = `${data.output_url}?v=${Date.now()}`;
    if (afterHint) afterHint.style.display = "none";

    if (data.lq_output_url) {
      if (lqCard) lqCard.hidden = false;
      if (lqPreview) lqPreview.src = `${data.lq_output_url}?v=${Date.now()}`;
      if (lqHint) lqHint.style.display = "none";
    } else {
      resetLqPreviewState();
    }

    const timeText = typeof data.elapsed_ms === "number" ? `，耗时 ${data.elapsed_ms} ms` : "";
    const devText = data.device ? `（${data.device}）` : "";
    setStatus(singleStatus, `完成：${data.model}${devText}${timeText}`, "ok");
    setStatus(singleRunInfo, `本次运行：run_id=${data.run_id || "-"}，流程=${data.workflow_mode || workflowMode}`);

    const inSize = data.input_size ? `${data.input_size.width}x${data.input_size.height}` : "-";
    const lqSize = data.lq_size ? `${data.lq_size.width}x${data.lq_size.height}` : "-";
    const outSize = data.output_size ? `${data.output_size.width}x${data.output_size.height}` : "-";
    if (data.workflow_mode === REAL_TO_LQ_WORKFLOW || workflowMode === REAL_TO_LQ_WORKFLOW) {
      setStatus(singleSizeInfo, `输入/低质/输出尺寸：${inSize} -> ${lqSize} -> ${outSize}`);
    } else {
      setStatus(singleSizeInfo, `输入/输出尺寸：${inSize} -> ${outSize}`);
    }

    const pathLines = [`输出：${data.saved_to}`];
    if (data.lq_saved_to) {
      pathLines.push(`低质量图：${data.lq_saved_to}`);
    }
    singleOutputPath.textContent = pathLines.join(" | ");

    lastSingleResultUrl = data.output_url;
    lastSingleSavedPath = data.saved_to || "";
    openResultBtn.disabled = false;
    copyResultPathBtn.disabled = false;

    singleDoneMetric.textContent = String(Number(singleDoneMetric.textContent || "0") + 1);
    addLog(`单图推理完成，run_id=${data.run_id || "-"}，流程=${data.workflow_mode || workflowMode}`, "success");
  } catch (err) {
    setStatus(singleStatus, `失败：${err.message}`, "error");
    setStatus(singleRunInfo, "本次运行：失败");
    setStatus(singleSizeInfo, workflowMode === REAL_TO_LQ_WORKFLOW ? "输入/低质/输出尺寸：-" : "输入/输出尺寸：-");
    if (workflowMode === REAL_TO_LQ_WORKFLOW && lqHint) {
      lqHint.textContent = "低质量图生成失败";
      lqHint.style.display = "block";
    }
    if (afterHint) {
      afterHint.textContent = "处理失败，请重试";
      afterHint.style.display = "block";
    }
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
  await copyText(lastSingleSavedPath, "已复制结果路径");
  setStatus(singleStatus, "已复制结果路径", "ok");
});

folderPickBtn.addEventListener("click", () => folderInput.click());
folderInput.addEventListener("change", () => {
  const files = Array.from(folderInput.files || []);
  if (!files.length) {
    setStatus(folderInfo, "尚未选择文件夹");
    return;
  }
  const root = files[0].webkitRelativePath && files[0].webkitRelativePath.includes("/")
    ? files[0].webkitRelativePath.split("/")[0]
    : "(无目录名)";
  setStatus(folderInfo, `已选择 ${files.length} 个文件，目录：${root}`);
  addLog(`已选择批量目录 ${root}，共 ${files.length} 个文件`);
});

const applyBatchResult = (data) => {
  const elapsedText = typeof data.elapsed_ms === "number" ? `${data.elapsed_ms} ms` : "-";
  setStatus(batchRunInfo, `本次运行：run_id=${data.run_id || "-"}`);
  setStatus(batchTimeInfo, `总耗时：${elapsedText}`);
  batchSummary.textContent = `处理成功 ${data.processed} 张，失败 ${data.failed || 0} 张，跳过 ${data.skipped} 张。`;
  batchOutputPath.textContent = `输出目录：${data.output_dir || "-"}`;

  batchPreviewGrid.innerHTML = "";
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

  batchErrorList.innerHTML = "";
  if (Array.isArray(data.errors) && data.errors.length) {
    data.errors.forEach((err) => {
      const li = document.createElement("li");
      li.textContent = err;
      batchErrorList.appendChild(li);
    });
  }
};

const fetchBatchStatus = async (runId) => {
  const resp = await fetch(`/api/process-batch/status/${encodeURIComponent(runId)}`);
  const data = await resp.json();
  if (!resp.ok || !data.ok) {
    throw new Error(data.error || "无法获取批量任务状态");
  }
  return data;
};

const onBatchTaskDone = (data) => {
  stopBatchPolling();
  batchRunBtn.disabled = false;
  batchCancelBtn.disabled = true;

  applyBatchResult(data);

  if (data.status === "success") {
    setStatus(batchStatus, `完成：${data.model_label || data.model_key}`, "ok");
    addLog(
      `批量处理完成：成功 ${data.processed}，失败 ${data.failed || 0}，跳过 ${data.skipped}，耗时 ${data.elapsed_ms ?? "-"} ms`,
      "success"
    );
  } else if (data.status === "canceled") {
    setStatus(batchStatus, "任务已取消", "error");
    addLog(`批量任务已取消，run_id=${data.run_id}`, "error");
  } else {
    setStatus(batchStatus, "任务失败", "error");
    addLog(`批量任务失败，run_id=${data.run_id}`, "error");
  }

  if (metricUpdatedForBatchRun !== data.run_id) {
    batchDoneMetric.textContent = String(
      Number(batchDoneMetric.textContent || "0") + Number(data.processed || 0)
    );
    metricUpdatedForBatchRun = data.run_id || "";
  }
  currentBatchRunId = "";
};

const pollBatchTask = async () => {
  if (!currentBatchRunId) return;
  try {
    const data = await fetchBatchStatus(currentBatchRunId);
    setBatchProgress(data.processed || 0, data.skipped || 0, data.failed || 0, data.total_uploaded || 0);

    const elapsedS = Math.max(0, Math.floor((Date.now() - currentBatchStartedAt) / 1000));
    setStatus(
      batchStatus,
      `处理中：${(data.processed || 0) + (data.skipped || 0) + (data.failed || 0)}/${data.total_uploaded || 0}，已运行 ${elapsedS}s`
    );

    if (data.done) {
      onBatchTaskDone(data);
    }
  } catch (err) {
    stopBatchPolling();
    batchRunBtn.disabled = false;
    batchCancelBtn.disabled = true;
    setStatus(batchStatus, `轮询失败：${err.message}`, "error");
    addLog(`批量任务轮询失败：${err.message}`, "error");
  }
};

batchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = Array.from(folderInput.files || []);
  if (!files.length) {
    setStatus(batchStatus, "请先选择文件夹。", "error");
    addLog("批量处理失败：未选择文件夹", "error");
    return;
  }

  let tile;
  try {
    tile = parseTile();
  } catch (err) {
    setStatus(batchStatus, err.message, "error");
    addLog(`批量处理失败：${err.message}`, "error");
    return;
  }

  stopBatchPolling();
  resetBatchResult();
  setBatchProgress(0, 0, 0, files.length);
  batchRunBtn.disabled = true;
  batchCancelBtn.disabled = false;
  setStatus(batchStatus, "任务提交中...");
  setStatus(batchRunInfo, "本次运行：任务准备中...");
  setStatus(batchTimeInfo, "总耗时：计算中...");
  addLog(`开始批量处理，共 ${files.length} 个上传项`);

  const formData = new FormData();
  formData.append("model_key", modelSelect.value);
  formData.append("tile", tile);
  files.forEach((file) => formData.append("files", file, file.webkitRelativePath || file.name));

  try {
    const resp = await fetch("/api/process-batch/start", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || "任务提交失败");
    }
    currentBatchRunId = data.run_id;
    currentBatchStartedAt = Date.now();
    setStatus(batchRunInfo, `本次运行：run_id=${data.run_id}`);
    setStatus(batchStatus, "任务已启动，正在处理中...");
    addLog(`批量任务已启动，run_id=${data.run_id}`);

    await pollBatchTask();
    if (currentBatchRunId) {
      batchPollTimer = setInterval(pollBatchTask, 1200);
    }
  } catch (err) {
    batchRunBtn.disabled = false;
    batchCancelBtn.disabled = true;
    setStatus(batchStatus, `失败：${err.message}`, "error");
    setStatus(batchRunInfo, "本次运行：失败");
    setStatus(batchTimeInfo, "总耗时：-");
    addLog(`批量处理失败：${err.message}`, "error");
  }
});

batchCancelBtn.addEventListener("click", async () => {
  if (!currentBatchRunId) return;
  batchCancelBtn.disabled = true;
  setStatus(batchStatus, "取消请求已发送，等待任务停止...");
  try {
    const resp = await fetch(`/api/process-batch/cancel/${encodeURIComponent(currentBatchRunId)}`, {
      method: "POST",
    });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || "取消失败");
    }
    addLog(`已请求取消任务 ${currentBatchRunId}`);
  } catch (err) {
    setStatus(batchStatus, `取消失败：${err.message}`, "error");
    addLog(`取消任务失败：${err.message}`, "error");
    batchCancelBtn.disabled = false;
  }
});

const renderDbLogs = (records) => {
  dbLogList.innerHTML = "";

  if (!Array.isArray(records) || records.length === 0) {
    const empty = document.createElement("li");
    empty.className = "db-log-empty";
    empty.textContent = "未查询到记录";
    dbLogList.appendChild(empty);
    return;
  }

  records.forEach((record) => {
    const item = document.createElement("li");
    item.className = `db-log-item is-${record.status || "unknown"}`;

    const top = document.createElement("div");
    top.className = "db-log-top";

    const runId = document.createElement("strong");
    runId.textContent = `run_id: ${record.run_id || "-"}`;
    top.appendChild(runId);

    const topActions = document.createElement("div");
    topActions.className = "db-log-actions";

    const badge = document.createElement("span");
    badge.className = `db-log-badge is-${record.status || "unknown"}`;
    badge.textContent = `${record.mode || "-"} / ${record.status || "-"}`;
    topActions.appendChild(badge);

    const copyRunIdBtn = document.createElement("button");
    copyRunIdBtn.type = "button";
    copyRunIdBtn.className = "db-mini-btn";
    copyRunIdBtn.textContent = "复制 run_id";
    copyRunIdBtn.addEventListener("click", () => copyText(record.run_id, "已复制 run_id"));
    topActions.appendChild(copyRunIdBtn);

    top.appendChild(topActions);

    const elapsed = typeof record.elapsed_ms === "number" ? `${record.elapsed_ms} ms` : "-";
    const meta = document.createElement("p");
    meta.className = "db-log-meta";
    meta.textContent = `${formatDateTime(record.created_at)} | model=${record.model_key || "-"} | elapsed=${elapsed}`;

    const detail = document.createElement("p");
    detail.className = "db-log-detail";
    detail.textContent = `processed=${record.processed ?? "-"}, skipped=${record.skipped ?? "-"}, output=${record.output_path || record.output_dir || "-"}`;

    item.appendChild(top);
    item.appendChild(meta);
    item.appendChild(detail);

    const outputPath = record.output_path || record.output_dir || "";
    if (outputPath) {
      const copyPathBtn = document.createElement("button");
      copyPathBtn.type = "button";
      copyPathBtn.className = "db-mini-btn";
      copyPathBtn.textContent = "复制输出路径";
      copyPathBtn.addEventListener("click", () => copyText(outputPath, "已复制输出路径"));
      item.appendChild(copyPathBtn);
    }

    if (record.error_message) {
      const errorDetails = document.createElement("details");
      errorDetails.className = "db-log-error-box";
      const summary = document.createElement("summary");
      summary.textContent = "查看错误信息";
      const err = document.createElement("p");
      err.className = "db-log-error";
      err.textContent = record.error_message;
      errorDetails.appendChild(summary);
      errorDetails.appendChild(err);
      item.appendChild(errorDetails);
    }

    dbLogList.appendChild(item);
  });
};

const updateLogPagination = () => {
  const page = Math.floor(logQueryState.offset / logQueryState.limit) + 1;
  const totalPages = Math.max(1, Math.ceil((logQueryState.total || 0) / logQueryState.limit));
  logsPageInfo.textContent = `第 ${page} / ${totalPages} 页`;
  logsPrevBtn.disabled = logQueryState.offset <= 0;
  logsNextBtn.disabled = (logQueryState.offset + logQueryState.limit) >= (logQueryState.total || 0);
};

const loadDbLogs = async () => {
  setStatus(dbLogStatus, "正在查询 SQLite 记录...");
  const params = new URLSearchParams();
  if (logQueryState.runId) params.set("run_id", logQueryState.runId);
  if (logQueryState.mode) params.set("mode", logQueryState.mode);
  if (logQueryState.status) params.set("status", logQueryState.status);
  if (logQueryState.createdFrom) params.set("created_from", logQueryState.createdFrom);
  if (logQueryState.createdTo) params.set("created_to", logQueryState.createdTo);
  params.set("limit", String(logQueryState.limit));
  params.set("offset", String(logQueryState.offset));

  try {
    const resp = await fetch(`/api/logs?${params.toString()}`);
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || "查询失败");
    }
    logQueryState.total = Number(data.total || 0);
    renderDbLogs(data.records || []);
    updateLogPagination();
    setStatus(dbLogStatus, `查询完成，共 ${data.total ?? 0} 条记录`, "ok");
    addLog(`SQLite 日志查询完成，返回 ${data.count ?? 0} 条`, "success");
  } catch (err) {
    renderDbLogs([]);
    logQueryState.total = 0;
    updateLogPagination();
    setStatus(dbLogStatus, `查询失败：${err.message}`, "error");
    addLog(`SQLite 日志查询失败：${err.message}`, "error");
  }
};

const refreshLogFilters = (resetOffset = true) => {
  logQueryState.runId = logRunIdInput.value.trim();
  logQueryState.mode = logModeSelect.value;
  logQueryState.status = logStatusSelect.value;
  logQueryState.createdFrom = logFromInput.value;
  logQueryState.createdTo = logToInput.value;
  logQueryState.limit = Number(logLimitSelect.value || 50);
  if (resetOffset) logQueryState.offset = 0;
};

clearLogBtn.addEventListener("click", () => {
  activityLog.innerHTML = "";
  addLog("会话日志已清空");
});

queryLogsBtn.addEventListener("click", () => {
  refreshLogFilters(true);
  loadDbLogs();
});

latestLogsBtn.addEventListener("click", () => {
  logRunIdInput.value = "";
  logModeSelect.value = "";
  logStatusSelect.value = "";
  logFromInput.value = "";
  logToInput.value = "";
  logLimitSelect.value = "50";
  refreshLogFilters(true);
  loadDbLogs();
});

clearDbViewBtn.addEventListener("click", () => {
  dbLogList.innerHTML = "";
  setStatus(dbLogStatus, "已清空查询结果列表");
  logQueryState.total = 0;
  logQueryState.offset = 0;
  updateLogPagination();
});

logsPrevBtn.addEventListener("click", () => {
  if (logQueryState.offset <= 0) return;
  logQueryState.offset = Math.max(0, logQueryState.offset - logQueryState.limit);
  loadDbLogs();
});

logsNextBtn.addEventListener("click", () => {
  if ((logQueryState.offset + logQueryState.limit) >= (logQueryState.total || 0)) return;
  logQueryState.offset += logQueryState.limit;
  loadDbLogs();
});

const bootstrap = async () => {
  applySingleWorkflowUI(getSingleWorkflowMode());

  try {
    const resp = await fetch("/api/models");
    const data = await resp.json();
    if (data.ok) {
      deviceBadge.textContent = `推理设备：${data.device}`;
      lqGeneratorReady = Boolean(data.lq_generator_ready);

      if (singleWorkflowSelect && Array.isArray(data.workflows) && data.workflows.length > 0) {
        const currentMode = singleWorkflowSelect.value || "";
        singleWorkflowSelect.innerHTML = "";
        data.workflows.forEach((wf) => {
          const option = document.createElement("option");
          option.value = wf.key;
          option.textContent = wf.label;
          singleWorkflowSelect.appendChild(option);
        });

        const defaultMode = data.default_workflow || currentMode || DIRECT_WORKFLOW;
        singleWorkflowSelect.value = defaultMode;
        if (!singleWorkflowSelect.value && data.workflows[0]?.key) {
          singleWorkflowSelect.value = data.workflows[0].key;
        }

        const realWorkflowOption = singleWorkflowSelect.querySelector(`option[value="${REAL_TO_LQ_WORKFLOW}"]`);
        if (realWorkflowOption) {
          realWorkflowOption.disabled = !lqGeneratorReady;
        }
        if (!lqGeneratorReady && singleWorkflowSelect.value === REAL_TO_LQ_WORKFLOW) {
          singleWorkflowSelect.value = DIRECT_WORKFLOW;
        }
      }

      applySingleWorkflowUI(getSingleWorkflowMode());
      setStatus(singleSizeInfo, getSingleWorkflowMode() === REAL_TO_LQ_WORKFLOW ? "输入/低质/输出尺寸：-" : "输入/输出尺寸：-");
      addLog(`初始化完成，当前设备 ${data.device}`, "success");
      if (!lqGeneratorReady) {
        addLog("lq_generator 未就绪，已禁用“真实图 -> 生成低质图 -> 超分”流程", "error");
      }
    } else {
      deviceBadge.textContent = "推理设备：未知";
      lqGeneratorReady = false;
      if (singleWorkflowSelect) singleWorkflowSelect.value = DIRECT_WORKFLOW;
      applySingleWorkflowUI(getSingleWorkflowMode());
      addLog("初始化失败：设备未知", "error");
    }
  } catch {
    deviceBadge.textContent = "推理设备：接口不可用";
    lqGeneratorReady = false;
    if (singleWorkflowSelect) singleWorkflowSelect.value = DIRECT_WORKFLOW;
    applySingleWorkflowUI(getSingleWorkflowMode());
    addLog("初始化失败：接口不可用", "error");
  }

  if (!("webkitdirectory" in folderInput)) {
    folderInput.removeAttribute("webkitdirectory");
    folderInput.removeAttribute("directory");
    batchCompatHint.textContent = "当前浏览器不支持目录选择，已自动降级为多文件选择。建议使用 Chrome/Edge。";
    addLog("浏览器不支持目录选择，已降级为多文件选择", "error");
  } else {
    batchCompatHint.textContent = "推荐使用 Chrome/Edge。目录上传在部分浏览器中可能不可用。";
  }

  refreshLogFilters(true);
  await loadDbLogs();
  updateLogPagination();
  if (dbLogDetails) dbLogDetails.open = false;
};

bootstrap();
