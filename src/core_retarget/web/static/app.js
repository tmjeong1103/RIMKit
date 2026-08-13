"use strict";

const state = {
  file: null,
  validation: null,
  robot: null,
  robots: [],
  jobId: null,
  eventSource: null,
  running: false,
  loadingExample: false,
};

const elements = {
  backendBadge: document.querySelector("#backend-badge"),
  motionInput: document.querySelector("#motion-input"),
  dropZone: document.querySelector("#drop-zone"),
  sourceChooser: document.querySelector("#source-chooser"),
  examplePicker: document.querySelector("#example-picker"),
  exampleGrid: document.querySelector("#example-grid"),
  exampleMessage: document.querySelector("#example-message"),
  chooserDivider: document.querySelector("#chooser-divider"),
  fileCard: document.querySelector("#file-card"),
  fileType: document.querySelector("#file-type"),
  fileName: document.querySelector("#file-name"),
  fileSize: document.querySelector("#file-size"),
  replaceFile: document.querySelector("#replace-file"),
  validationBadge: document.querySelector("#validation-badge"),
  validationMessage: document.querySelector("#validation-message"),
  statFrames: document.querySelector("#stat-frames"),
  statFps: document.querySelector("#stat-fps"),
  statDuration: document.querySelector("#stat-duration"),
  statContacts: document.querySelector("#stat-contacts"),
  robotSelect: document.querySelector("#robot-select"),
  selectedRobotName: document.querySelector("#selected-robot-name"),
  selectedRobotManufacturer: document.querySelector("#selected-robot-manufacturer"),
  selectedRobotDof: document.querySelector("#selected-robot-dof"),
  fpsInput: document.querySelector("#fps-input"),
  resolutionInput: document.querySelector("#resolution-input"),
  saveStagesInput: document.querySelector("#save-stages-input"),
  stageArchivesRow: document.querySelector("#stage-archives-row"),
  deploymentNote: document.querySelector("#deployment-note"),
  runButton: document.querySelector("#run-button"),
  formError: document.querySelector("#form-error"),
  resultTitle: document.querySelector("#result-title"),
  statusPill: document.querySelector("#status-pill"),
  previewPlaceholder: document.querySelector("#preview-placeholder"),
  resultVideo: document.querySelector("#result-video"),
  progressLabel: document.querySelector("#progress-label"),
  progressPercent: document.querySelector("#progress-percent"),
  progressBar: document.querySelector("#progress-bar"),
  pipelineSteps: document.querySelector("#pipeline-steps"),
  activityToggle: document.querySelector("#activity-toggle"),
  activityBody: document.querySelector("#activity-body"),
  activityCurrent: document.querySelector("#activity-current"),
  frameProgress: document.querySelector("#frame-progress"),
  frameLabel: document.querySelector("#frame-label"),
  frameBar: document.querySelector("#frame-bar"),
  eventLog: document.querySelector("#event-log"),
  downloads: document.querySelector("#downloads"),
  motionDownload: document.querySelector("#motion-download"),
  videoDownload: document.querySelector("#video-download"),
  manifestDownload: document.querySelector("#manifest-download"),
};

async function loadBackend() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) return;
    const payload = await response.json();
    const backend = payload.backend === "native" ? "Native backend" : "Python backend";
    const limits = payload.limits || {};
    const hosted = limits.max_active_jobs !== null || limits.result_ttl_seconds !== null;
    elements.backendBadge.lastChild.textContent = ` ${hosted ? "Public demo" : "Local"} · ${backend}`;

    if (limits.allow_stage_archives === false) {
      elements.saveStagesInput.checked = false;
      elements.saveStagesInput.disabled = true;
      elements.stageArchivesRow.classList.add("hidden");
    }

    const maxWidth = Number(limits.max_video_width || Number.MAX_SAFE_INTEGER);
    const maxHeight = Number(limits.max_video_height || Number.MAX_SAFE_INTEGER);
    let firstAllowed = null;
    Array.from(elements.resolutionInput.options).forEach((option) => {
      const [width, height] = option.value.split("x").map(Number);
      const allowed = width <= maxWidth && height <= maxHeight;
      option.disabled = !allowed;
      option.hidden = !allowed;
      if (allowed && firstAllowed === null) firstAllowed = option.value;
    });
    if (elements.resolutionInput.selectedOptions[0]?.disabled && firstAllowed !== null) {
      elements.resolutionInput.value = firstAllowed;
    }

    if (limits.result_ttl_seconds !== null && limits.result_ttl_seconds !== undefined) {
      const minutes = Math.round(Number(limits.result_ttl_seconds) / 60);
      elements.deploymentNote.textContent = `Validated by CoRe · uploads and results expire after ${minutes} min`;
    }
  } catch (_error) {
    // The regular health and job requests will expose an unavailable server.
  }
}

const pipelineGroups = ["validation", "dmr", "refinement", "collision", "render", "export"];
const stageGroups = {
  CREATED: "validation",
  VALIDATING: "validation",
  LOADING_MODEL: "validation",
  DMR: "dmr",
  INITIAL_COLLISION: "refinement",
  EXTRACTING_TRAJECTORIES: "refinement",
  ARA: "refinement",
  FPA_TARGETS: "refinement",
  FPA_IK: "refinement",
  GROUNDING: "refinement",
  FINAL_COLLISION: "collision",
  VALIDATING_OUTPUT: "render",
  EXPORTING: "export",
  RENDERING: "render",
  SUCCEEDED: "export",
  SUCCEEDED_WITH_WARNINGS: "export",
};

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function sourceFormat(file, validation = null) {
  const validated = String(validation?.container_format || "").trim().toUpperCase();
  if (validated) return validated;
  const suffix = file.name.split(".").pop()?.trim().toUpperCase();
  return suffix === "NPZ" || suffix === "PT" ? suffix : "SOMA";
}

function sourceProvider(validation) {
  const provider = String(validation?.provider || "").trim().toLowerCase();
  if (provider === "gem-x" || provider === "gemx") return "GEM-X";
  if (provider === "kimodo") return "Kimodo";
  return "SOMA";
}

function setText(element, value) {
  element.textContent = value;
}

function setStatus(status, label) {
  elements.statusPill.className = `status-pill ${status}`;
  setText(elements.statusPill, label || status);
}

function errorDetail(response, fallback) {
  return response.json()
    .then((payload) => payload.detail || fallback)
    .catch(() => fallback);
}

function refreshRunButton() {
  elements.runButton.disabled = !state.file || !state.validation || !state.robot || state.running;
  elements.replaceFile.disabled = state.running;
  elements.robotSelect.disabled = state.running || state.robots.length === 0;
  for (const button of elements.exampleGrid.querySelectorAll("button")) {
    button.disabled = state.running || state.loadingExample;
  }
}

function resetResult() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.jobId = null;
  elements.resultVideo.pause();
  elements.resultVideo.removeAttribute("src");
  elements.resultVideo.load();
  elements.resultVideo.classList.add("hidden");
  elements.previewPlaceholder.classList.remove("hidden");
  elements.downloads.classList.add("hidden");
  elements.eventLog.replaceChildren();
  elements.progressBar.style.width = "0%";
  setText(elements.progressPercent, "0%");
  setText(elements.progressLabel, "Pipeline progress");
  setText(elements.activityCurrent, "Waiting to start.");
  setText(elements.resultTitle, "Ready when you are");
  setStatus("idle", "Idle");
  elements.frameProgress.classList.add("hidden");
  for (const item of elements.pipelineSteps.querySelectorAll("li")) {
    item.classList.remove("active", "complete");
  }
}

function showFile(file) {
  if (state.running) return;
  if (state.jobId) resetResult();
  state.file = file;
  state.validation = null;
  elements.fileCard.classList.remove("hidden");
  elements.sourceChooser.classList.add("hidden");
  setText(elements.fileType, sourceFormat(file));
  setText(elements.fileName, file.name);
  setText(elements.fileSize, formatBytes(file.size));
  elements.validationBadge.className = "validation-badge";
  setText(elements.validationBadge, "Checking");
  elements.validationMessage.classList.add("hidden");
  for (const stat of [
    elements.statFrames,
    elements.statFps,
    elements.statDuration,
    elements.statContacts,
  ]) {
    setText(stat, "—");
  }
  refreshRunButton();
  validateFile(file);
}

function showSourceChooser() {
  if (state.running) return;
  state.file = null;
  state.validation = null;
  elements.motionInput.value = "";
  elements.fileCard.classList.add("hidden");
  elements.sourceChooser.classList.remove("hidden");
  elements.validationMessage.classList.add("hidden");
  refreshRunButton();
}

function exampleButton(example) {
  const button = document.createElement("button");
  button.className = "example-motion";
  button.type = "button";
  button.dataset.testid = `example-${example.id}`;

  const format = document.createElement("span");
  format.className = "example-format";
  format.textContent = String(example.format || "SOMA").toUpperCase();

  const copy = document.createElement("span");
  copy.className = "example-copy";
  const provider = document.createElement("small");
  provider.textContent = example.provider;
  const title = document.createElement("strong");
  title.textContent = example.title;
  const filename = document.createElement("span");
  filename.textContent = `${example.filename} · ${formatBytes(example.size_bytes)}`;
  copy.append(provider, title, filename);

  const arrow = document.createElement("span");
  arrow.className = "example-arrow";
  arrow.setAttribute("aria-hidden", "true");
  arrow.textContent = "→";
  button.append(format, copy, arrow);
  button.addEventListener("click", () => loadExampleMotion(example, button));
  return button;
}

async function loadExampleMotion(example, button) {
  if (state.running || state.loadingExample) return;
  state.loadingExample = true;
  elements.exampleMessage.classList.add("hidden");
  button.classList.add("loading");
  refreshRunButton();
  try {
    const response = await fetch(example.url);
    if (!response.ok) throw new Error(await errorDetail(response, "Example motion is unavailable."));
    const blob = await response.blob();
    const file = new File([blob], example.filename, { type: "application/octet-stream" });
    showFile(file);
  } catch (error) {
    setText(elements.exampleMessage, error.message);
    elements.exampleMessage.classList.remove("hidden");
  } finally {
    state.loadingExample = false;
    button.classList.remove("loading");
    refreshRunButton();
  }
}

async function loadExampleMotions() {
  try {
    const response = await fetch("/api/motions/examples");
    if (!response.ok) throw new Error("Bundled examples could not be loaded.");
    const payload = await response.json();
    const examples = Array.isArray(payload.examples) ? payload.examples : [];
    if (examples.length === 0) {
      elements.examplePicker.classList.add("hidden");
      elements.chooserDivider.classList.add("hidden");
      return;
    }
    elements.exampleGrid.replaceChildren(...examples.map(exampleButton));
  } catch (_error) {
    elements.examplePicker.classList.add("hidden");
    elements.chooserDivider.classList.add("hidden");
  }
}

async function validateFile(file) {
  const form = new FormData();
  form.append("motion", file);
  const fps = elements.fpsInput.value.trim();
  if (fps) form.append("fps", fps);
  try {
    const response = await fetch("/api/motions/validate", { method: "POST", body: form });
    if (!response.ok) throw new Error(await errorDetail(response, "Motion validation failed."));
    const payload = await response.json();
    if (file !== state.file) return;
    state.validation = payload;
    setText(elements.fileType, sourceFormat(file, payload));
    elements.validationBadge.className = "validation-badge valid";
    setText(elements.validationBadge, `Valid ${sourceProvider(payload)}`);
    setText(elements.statFrames, payload.frame_count.toLocaleString());
    setText(elements.statFps, Number(payload.fps).toFixed(2));
    setText(elements.statDuration, `${Number(payload.duration_seconds).toFixed(2)} s`);
    setText(
      elements.statContacts,
      payload.contact_channels === null ? "Derived" : `${payload.contact_channels} ch`,
    );
    if (Array.isArray(payload.warnings) && payload.warnings.length) {
      setText(elements.validationMessage, payload.warnings.join(" "));
      elements.validationMessage.classList.remove("hidden");
      elements.validationMessage.style.color = "#76600c";
    }
  } catch (error) {
    if (file !== state.file) return;
    state.validation = null;
    elements.validationBadge.className = "validation-badge invalid";
    setText(elements.validationBadge, "Invalid");
    setText(elements.validationMessage, error.message);
    elements.validationMessage.classList.remove("hidden");
    elements.validationMessage.style.color = "";
  } finally {
    refreshRunButton();
  }
}

function findRobot(robotId = state.robot) {
  return state.robots.find((robot) => robot.id === robotId) || null;
}

function robotName(robotId = state.robot) {
  return findRobot(robotId)?.name || robotId?.toUpperCase() || "Robot";
}

function updateRobotSelection() {
  const robot = findRobot();
  if (!robot) {
    setText(elements.selectedRobotName, "Unavailable");
    setText(elements.selectedRobotManufacturer, "Robot models could not be loaded.");
    setText(elements.selectedRobotDof, "— DOF");
    return;
  }
  setText(elements.selectedRobotName, robot.name);
  setText(elements.selectedRobotManufacturer, robot.manufacturer);
  setText(elements.selectedRobotDof, `${robot.dof} DOF`);
}

async function loadRobots() {
  try {
    const response = await fetch("/api/robots");
    if (!response.ok) throw new Error("Robot models could not be loaded.");
    const payload = await response.json();
    if (!Array.isArray(payload.robots) || payload.robots.length === 0) {
      throw new Error("No robot models are available.");
    }

    state.robots = payload.robots;
    elements.robotSelect.replaceChildren();
    state.robots.forEach((robot) => {
      const option = document.createElement("option");
      option.value = robot.id;
      option.textContent = `${robot.name} · ${robot.manufacturer} · ${robot.dof} DOF`;
      elements.robotSelect.append(option);
    });
    state.robot = state.robots[0].id;
    elements.robotSelect.value = state.robot;
    updateRobotSelection();
  } catch (error) {
    state.robots = [];
    state.robot = null;
    const option = document.createElement("option");
    option.value = "";
    option.textContent = error.message;
    elements.robotSelect.replaceChildren(option);
    updateRobotSelection();
  } finally {
    refreshRunButton();
  }
}

function stageProgress(stage, current, total, complete) {
  const group = stageGroups[stage] || "validation";
  const groupIndex = pipelineGroups.indexOf(group);
  let within = 0.18;
  if (current !== null && total) within = Math.min(0.95, current / total);
  if (complete) within = 1;
  const percent = Math.round(((groupIndex + within) / pipelineGroups.length) * 100);
  elements.progressBar.style.width = `${percent}%`;
  setText(elements.progressPercent, `${percent}%`);

  for (const item of elements.pipelineSteps.querySelectorAll("li")) {
    const itemIndex = pipelineGroups.indexOf(item.dataset.group);
    item.classList.toggle("complete", itemIndex < groupIndex || (complete && itemIndex === groupIndex));
    item.classList.toggle("active", itemIndex === groupIndex && !complete);
  }
}

function appendEvent(event) {
  const line = document.createElement("li");
  const time = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "";
  line.textContent = `${time}  ${event.message || event.event_type}`;
  elements.eventLog.append(line);
  while (elements.eventLog.children.length > 30) {
    elements.eventLog.removeChild(elements.eventLog.firstElementChild);
  }
  elements.eventLog.scrollTop = elements.eventLog.scrollHeight;
}

function handleEvent(event) {
  appendEvent(event);
  if (event.event_type === "job_started") setStatus("running", "Running");
  if (event.message) setText(elements.activityCurrent, event.message);
  if (event.stage) {
    const isStageComplete = event.event_type === "stage_completed";
    stageProgress(event.stage, event.current ?? null, event.total ?? null, isStageComplete);
  }
  if (event.current !== null && event.current !== undefined && event.total) {
    elements.frameProgress.classList.remove("hidden");
    setText(elements.frameLabel, `${event.current} / ${event.total}`);
    elements.frameBar.style.width = `${Math.min(100, (event.current / event.total) * 100)}%`;
  } else {
    elements.frameProgress.classList.add("hidden");
  }
}

async function refreshJob() {
  const response = await fetch(`/api/jobs/${state.jobId}`);
  if (!response.ok) throw new Error(await errorDetail(response, "Job status is unavailable."));
  const job = await response.json();
  if (job.status === "succeeded") {
    state.running = false;
    setStatus("succeeded", "Complete");
    setText(elements.resultTitle, `${robotName(job.robot_id)} motion is ready`);
    setText(elements.progressPercent, "100%");
    elements.progressBar.style.width = "100%";
    for (const item of elements.pipelineSteps.querySelectorAll("li")) {
      item.classList.remove("active");
      item.classList.add("complete");
    }
    showArtifacts(job.artifacts);
  } else if (job.status === "failed") {
    state.running = false;
    setStatus("failed", "Failed");
    setText(elements.resultTitle, "Retargeting could not finish");
    showFormError(job.error || "The CoRe pipeline failed.");
  }
  refreshRunButton();
  return job;
}

function showArtifacts(artifacts) {
  if (artifacts.video) {
    elements.resultVideo.src = artifacts.video.url;
    elements.resultVideo.classList.remove("hidden");
    elements.previewPlaceholder.classList.add("hidden");
    elements.videoDownload.href = artifacts.video.url;
    elements.videoDownload.classList.remove("hidden");
  } else {
    elements.videoDownload.classList.add("hidden");
  }
  if (artifacts.motion) elements.motionDownload.href = artifacts.motion.url;
  if (artifacts.manifest) elements.manifestDownload.href = artifacts.manifest.url;
  elements.downloads.classList.remove("hidden");
}

function listenForEvents(jobId) {
  if (state.eventSource) state.eventSource.close();
  const source = new EventSource(`/api/jobs/${jobId}/events`);
  state.eventSource = source;
  source.onmessage = async (message) => {
    const event = JSON.parse(message.data);
    handleEvent(event);
    if (event.event_type === "job_succeeded" || event.event_type === "job_failed") {
      source.close();
      state.eventSource = null;
      try {
        await refreshJob();
      } catch (error) {
        showFormError(error.message);
      }
    }
  };
  source.onerror = async () => {
    source.close();
    state.eventSource = null;
    try {
      const job = await refreshJob();
      if (job.status === "queued" || job.status === "running") {
        window.setTimeout(() => listenForEvents(jobId), 1000);
      }
    } catch (error) {
      showFormError(error.message);
      state.running = false;
      refreshRunButton();
    }
  };
}

function showFormError(message) {
  setText(elements.formError, message);
  elements.formError.classList.remove("hidden");
}

async function runRetargeting() {
  if (!state.file || !state.validation || !state.robot || state.running) return;
  resetResult();
  elements.formError.classList.add("hidden");
  state.running = true;
  refreshRunButton();
  setStatus("running", "Queued");
  setText(elements.resultTitle, `Preparing ${robotName()}`);
  setText(elements.activityCurrent, "Uploading the validated source motion.");
  elements.activityBody.classList.remove("hidden");
  elements.activityToggle.setAttribute("aria-expanded", "true");

  const form = new FormData();
  form.append("motion", state.file);
  form.append("robot", state.robot);
  form.append("render_video", "true");
  form.append("save_stages", String(elements.saveStagesInput.checked));
  const fps = elements.fpsInput.value.trim();
  if (fps) form.append("fps", fps);
  const [width, height] = elements.resolutionInput.value.split("x");
  form.append("width", width);
  form.append("height", height);

  try {
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    if (!response.ok) throw new Error(await errorDetail(response, "Could not create a CoRe job."));
    const job = await response.json();
    state.jobId = job.job_id;
    setStatus("running", job.status === "queued" ? "Queued" : "Running");
    setText(elements.resultTitle, `${robotName()} retargeting in progress`);
    listenForEvents(job.job_id);
  } catch (error) {
    state.running = false;
    setStatus("failed", "Failed");
    setText(elements.resultTitle, "Could not start retargeting");
    showFormError(error.message);
    refreshRunButton();
  }
}

elements.motionInput.addEventListener("change", () => {
  const file = elements.motionInput.files?.[0];
  if (file) showFile(file);
});

elements.replaceFile.addEventListener("click", () => {
  showSourceChooser();
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
}

elements.dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (file) showFile(file);
});

elements.fpsInput.addEventListener("change", () => {
  if (state.file) validateFile(state.file);
});

elements.robotSelect.addEventListener("change", () => {
  state.robot = elements.robotSelect.value || null;
  updateRobotSelection();
  refreshRunButton();
});

elements.activityToggle.addEventListener("click", () => {
  const expanded = elements.activityToggle.getAttribute("aria-expanded") === "true";
  elements.activityToggle.setAttribute("aria-expanded", String(!expanded));
  elements.activityBody.classList.toggle("hidden", expanded);
});

elements.runButton.addEventListener("click", runRetargeting);

resetResult();
loadBackend();
loadRobots();
loadExampleMotions();
