const form = document.querySelector("#generatorForm");
const audioInput = document.querySelector("#audioInput");
const dropzone = document.querySelector("#dropzone");
const dropTitle = document.querySelector("#dropTitle");
const dropDetail = document.querySelector("#dropDetail");
const fileMeta = document.querySelector("#fileMeta");
const titleInput = document.querySelector("#titleInput");
const autoTiming = document.querySelector("#autoTiming");
const bpmInput = document.querySelector("#bpmInput");
const offsetInput = document.querySelector("#offsetInput");
const levelRange = document.querySelector("#levelRange");
const levelNumber = document.querySelector("#levelNumber");
const densityValue = document.querySelector("#densityValue");
const densitySamples = document.querySelector("#densitySamples");
const heatInputs = [...document.querySelectorAll(".heat-row input[type='range']")];
const generateButton = document.querySelector("#generateButton");
const formMessage = document.querySelector("#formMessage");
const idleState = document.querySelector("#idleState");
const workingState = document.querySelector("#workingState");
const workingDetail = document.querySelector("#workingDetail");
const resultState = document.querySelector("#resultState");
const elapsed = document.querySelector("#elapsed");
const historyList = document.querySelector("#historyList");
const serverState = document.querySelector("#serverState");
const stateRoot = document.querySelector(".server-state");

let selectedFile = null;
let timer = null;
let densityProfile = [];

function syncTimingMode() {
  bpmInput.disabled = autoTiming.checked;
  offsetInput.disabled = autoTiming.checked;
}

autoTiming.addEventListener("change", syncTimingMode);
syncTimingMode();

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
  return `${value.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function selectFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".mp3")) {
    formMessage.textContent = "请选择 MP3 文件";
    return;
  }
  selectedFile = file;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  audioInput.files = transfer.files;
  dropzone.classList.add("has-file");
  dropTitle.textContent = file.name;
  dropDetail.textContent = formatBytes(file.size);
  fileMeta.textContent = "已选择";
  if (!titleInput.value.trim()) titleInput.value = file.name.replace(/\.mp3$/i, "");
  formMessage.textContent = "";
}

audioInput.addEventListener("change", () => selectFile(audioInput.files[0]));
["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", event => selectFile(event.dataTransfer.files[0]));

function updateDensity() {
  if (!densityProfile.length) return;
  const level = Number(levelNumber.value);
  const row = densityProfile.reduce((best, item) => (
    Math.abs(item.level - level) < Math.abs(best.level - level) ? item : best
  ));
  densityValue.textContent = `${Number(row.rms_notes_per_second).toFixed(2)} notes/s RMS`;
  densitySamples.textContent = `${row.exact_charts} 张同级 / ${row.nearby_charts} 张邻近`;
}

function updateRangeVisual(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const value = Number(input.value);
  const percent = ((value - min) / Math.max(0.0001, max - min)) * 100;
  input.style.setProperty("--range-fill", `${Math.max(0, Math.min(100, percent))}%`);
}

function syncLevel(source) {
  const raw = Number(source.value);
  if (!Number.isFinite(raw)) return;
  const value = Math.max(12, Math.min(15, Math.round(raw * 10) / 10));
  levelRange.value = value.toFixed(1);
  levelNumber.value = value.toFixed(1);
  updateRangeVisual(levelRange);
  updateDensity();
}

levelRange.addEventListener("input", () => syncLevel(levelRange));
levelNumber.addEventListener("input", () => syncLevel(levelNumber));
levelNumber.addEventListener("change", () => syncLevel(levelNumber));
syncLevel(levelRange);

heatInputs.forEach(input => {
  const output = input.closest(".heat-row").querySelector("output");
  const update = () => {
    const text = `${Math.round(Number(input.value) * 100)}%`;
    output.value = text;
    output.textContent = text;
    updateRangeVisual(input);
  };
  input.addEventListener("input", update);
  update();
});

async function loadDensityProfile() {
  try {
    const response = await fetch("/api/density-profile", { cache: "no-store" });
    const payload = await response.json();
    densityProfile = payload.profile || [];
    updateDensity();
  } catch (_) {
    densityValue.textContent = "统计不可用";
    densitySamples.textContent = "";
  }
}

function setWorking(active) {
  generateButton.disabled = active;
  idleState.classList.toggle("hidden", active);
  workingState.classList.toggle("hidden", !active);
  if (active) resultState.classList.add("hidden");
}

function startTimer() {
  const started = Date.now();
  elapsed.textContent = "0.0s";
  timer = setInterval(() => { elapsed.textContent = `${((Date.now() - started) / 1000).toFixed(1)}s`; }, 100);
}

function stopTimer() {
  clearInterval(timer);
  timer = null;
}

function displayModelName(value) {
  return {
    "v1.7.1": "v1.7.1",
    "trans-02": "Trans-02",
    "v2": "ORBIT-8 v2",
    "v2-16m": "ORBIT-8 v2 16M",
    "v2.1-handflow": "ORBIT-8 v2.1 HandFlow",
  }[value] || value;
}

function renderResult(data) {
  const report = data.report || {};
  const types = report.event_types || {};
  const modelName = displayModelName(report.web_model || "v2.1-handflow");
  resultState.innerHTML = `
    <h3 class="result-title">${escapeHtml(data.folder_name)}</h3>
    <p class="model-result">${modelName}</p>
    <div class="stat-grid">
      <div class="stat"><strong>${report.events ?? "-"}</strong><span>Notes</span></div>
      <div class="stat"><strong>${types.slide ?? "-"}</strong><span>Slides</span></div>
      <div class="stat"><strong>${report.breaks ?? "-"}</strong><span>Breaks</span></div>
    </div>
    <p class="result-path">${escapeHtml(data.folder_path)}</p>
    <p class="timing-result">BPM ${report.bpm ?? "-"} · Offset ${report.offset_seconds ?? "-"}s${report.automatic_timing ? " · Auto" : ""}</p>
    <p class="timing-result">密度 ${report.density_calibration?.target_rms_notes_per_second?.toFixed?.(2) ?? "-"} notes/s RMS · 交互 ${Math.round((report.pattern_heat?.interaction ?? 1) * 100)}% · 扫键 ${Math.round((report.pattern_heat?.sweep ?? 1) * 100)}% · 纵连 ${Math.round((report.pattern_heat?.jack ?? 1) * 100)}%</p>
    <div class="file-actions">
      <a href="${data.files.maidata}" target="_blank">maidata.txt</a>
      <a href="${data.files.audio}" target="_blank">track.mp3</a>
      <a href="${data.files.report}" target="_blank">生成报告</a>
    </div>`;
  resultState.classList.remove("hidden");
  idleState.classList.add("hidden");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char]));
}

async function loadHistory() {
  try {
    const response = await fetch("/api/results");
    const payload = await response.json();
    if (!payload.results.length) {
      historyList.innerHTML = '<div class="history-empty">暂无输出</div>';
      return;
    }
    historyList.innerHTML = payload.results.map(item => {
      const report = item.report || {};
      const date = new Date(item.created_at * 1000);
      return `<div class="history-item">
        <strong>${escapeHtml(item.folder_name)}</strong>
        <time>${date.toLocaleString()}</time>
        <span>Lv ${report.level ?? "-"} · ${report.events ?? "-"} notes</span>
        <a href="${item.files.maidata}" target="_blank">打开</a>
      </div>`;
    }).join("");
  } catch (_) {
    historyList.innerHTML = '<div class="history-empty">读取失败</div>';
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (!selectedFile) {
    formMessage.textContent = "请选择 MP3 文件";
    return;
  }
  formMessage.textContent = "";
  setWorking(true);
  startTimer();
  const body = new FormData(form);
  body.set("audio", selectedFile, selectedFile.name);
  body.set("auto_timing", autoTiming.checked ? "true" : "false");
  const modelName = displayModelName(body.get("model"));
  workingDetail.textContent = `${modelName} · Lv ${levelNumber.value} · 交互 ${body.get("interaction_heat") * 100}% · 扫键 ${body.get("sweep_heat") * 100}% · 纵连 ${body.get("jack_heat") * 100}%`;
  try {
    const response = await fetch("/api/generate", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "生成失败");
    workingState.classList.add("hidden");
    renderResult(payload);
    await loadHistory();
  } catch (error) {
    workingState.classList.add("hidden");
    idleState.classList.remove("hidden");
    idleState.querySelector("p").textContent = "生成失败";
    formMessage.textContent = error.message;
  } finally {
    stopTimer();
    generateButton.disabled = false;
  }
});

document.querySelector("#refreshButton").addEventListener("click", loadHistory);

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    stateRoot.className = "server-state ready";
    serverState.textContent = payload.busy ? "GPU 忙碌" : "本地就绪";
  } catch (_) {
    stateRoot.className = "server-state error";
    serverState.textContent = "服务断开";
  }
}

checkHealth();
loadDensityProfile();
loadHistory();
setInterval(checkHealth, 4000);
