// Keep API requests on FastAPI even when the dashboard is previewed with
// Cursor/VS Code Live Server. The backend serves the page itself on port 8000.
const API_BASE = window.location.protocol === "http:" || window.location.protocol === "https:"
  ? window.location.origin
  : "http://127.0.0.1:8000";

const byId = (id) => document.getElementById(id);
let hasLoadedEvents = false;
let lastEventKey = null;
let refreshInProgress = false;
let previousEmergencySnapshot = null;
let previousCandidateSnapshot = null;
let hasFeaturedEmergency = false;
let hasFeaturedCandidate = false;

function updateFeaturedClipsVisibility() {
  const count = Number(hasFeaturedEmergency) + Number(hasFeaturedCandidate);
  byId("featuredEmergencyPanel").hidden = !hasFeaturedEmergency;
  byId("featuredCandidatePanel").hidden = !hasFeaturedCandidate;
  byId("featuredClips").hidden = count === 0;
  byId("featuredClips").classList.toggle("single", count === 1);
}

function formatDate(value, options = {}) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString(undefined, options);
}

function eventKey(event) {
  return String(event?.id ?? event?.timestamp ?? "");
}

async function fetchJSON(route) {
  const response = await fetch(`${API_BASE}${route}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`${route} returned HTTP ${response.status}`);
  return response.json();
}

function setBackendState(online, message = "") {
  const pill = byId("backendPill");
  pill.classList.toggle("connected", online);
  pill.classList.toggle("offline", !online);
  byId("backendLabel").textContent = online ? "Backend connected" : "Backend offline";
  byId("connectionBanner").hidden = online;
  if (!online) {
    byId("connectionMessage").textContent = message ||
      `Could not reach ${API_BASE}. Start the backend with: python -m uvicorn backend.server:app --host 127.0.0.1 --port 8000`;
  }
}

function makeEmpty(message) {
  const empty = document.createElement("div");
  empty.className = "empty-state-inline";
  empty.textContent = message;
  return empty;
}

function renderCameraGrid() {
  const cameras = [
    { id: "CAM 01", name: "Primary workstation camera", state: "Managed by desktop app", configured: true },
    { id: "CAM 02", name: "Additional camera source", state: "Source not configured", configured: false },
    { id: "CAM 03", name: "Additional camera source", state: "Source not configured", configured: false },
    { id: "CAM 04", name: "Additional camera source", state: "Source not configured", configured: false },
  ];
  const grid = byId("cameraGrid");
  grid.replaceChildren();
  for (const camera of cameras) {
    const card = document.createElement("article");
    card.className = "camera-card";
    const preview = document.createElement("div");
    preview.className = "camera-preview";
    const placeholder = document.createElement("div");
    placeholder.className = "camera-placeholder";
    const icon = document.createElement("span");
    icon.textContent = "▣";
    icon.setAttribute("aria-hidden", "true");
    const title = document.createElement("strong");
    title.textContent = camera.configured ? "Desktop camera · no web stream" : "No feed configured";
    const hint = document.createElement("small");
    hint.textContent = camera.configured ? "This dashboard does not receive the camera video" : "Add a camera source to enable this view";
    placeholder.append(icon, title, hint);
    preview.appendChild(placeholder);
    const body = document.createElement("div");
    body.className = "camera-card-body";
    const name = document.createElement("div");
    name.className = "camera-name";
    const id = document.createElement("strong");
    id.textContent = camera.id;
    const description = document.createElement("span");
    description.textContent = camera.name;
    name.append(id, description);
    const state = document.createElement("span");
    state.className = `camera-state${camera.configured ? " available" : ""}`;
    const dot = document.createElement("i");
    const stateText = document.createElement("span");
    stateText.textContent = camera.configured ? "NO WEB FEED" : "NOT SET UP";
    state.append(dot, stateText);
    body.append(name, state);
    card.append(preview, body);
    grid.appendChild(card);
  }
  byId("cameraCountBadge").textContent = "1 SOURCE · 3 TO CONFIGURE";
}

function makeClipCard({ url, kind, title, time, meta, score, candidate = false }) {
  const card = document.createElement("article");
  card.className = "clip-card";

  const video = document.createElement("video");
  video.className = "clip-card-video";
  video.controls = true;
  video.preload = "metadata";
  video.playsInline = true;
  video.src = url;
  video.setAttribute("aria-label", `${kind} video: ${title}`);
  card.appendChild(video);

  const body = document.createElement("div");
  body.className = "clip-card-body";
  const top = document.createElement("div");
  top.className = "clip-card-top";
  const kindLabel = document.createElement("span");
  kindLabel.className = `clip-kind${candidate ? " candidate-kind" : ""}`;
  kindLabel.textContent = `● ${kind}`;
  const timeLabel = document.createElement("span");
  timeLabel.className = "clip-card-time";
  timeLabel.textContent = formatDate(time, { dateStyle: "medium", timeStyle: "short" });
  top.append(kindLabel, timeLabel);

  const titleLabel = document.createElement("strong");
  titleLabel.className = "clip-card-title";
  titleLabel.title = title;
  titleLabel.textContent = title;

  const details = document.createElement("div");
  details.className = "clip-card-meta";
  const metaLabel = document.createElement("span");
  metaLabel.textContent = meta;
  details.appendChild(metaLabel);
  if (score !== undefined && score !== null && Number.isFinite(Number(score))) {
    const scoreLabel = document.createElement("span");
    scoreLabel.className = "candidate-score";
    scoreLabel.textContent = `Score ${(Number(score) * 100).toFixed(1)}%`;
    details.appendChild(scoreLabel);
  }
  body.append(top, titleLabel, details);
  if (candidate) {
    const review = document.createElement("span");
    review.className = "review-label";
    review.textContent = "PENDING HUMAN REVIEW";
    body.appendChild(review);
  }
  card.appendChild(body);
  return card;
}

function renderFeaturedRecording(recording) {
  const video = byId("featuredVideo");
  const empty = byId("featuredEmpty");
  const button = byId("openLatestButton");
  const badge = byId("latestRecordingBadge");
  const timestamp = byId("mediaTimestamp");
  hasFeaturedEmergency = Boolean(recording?.video);
  updateFeaturedClipsVisibility();

  if (!recording) {
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.hidden = true;
    empty.hidden = false;
    button.disabled = true;
    badge.textContent = "NO RECORDING";
    timestamp.hidden = true;
    byId("latestRecordedAt").textContent = "No recording";
    byId("latestAlertStatus").textContent = "No event selected";
    byId("latestAlertStatus").classList.add("text-muted");
    return;
  }

  const url = `${API_BASE}/videos/${encodeURIComponent(recording.video)}`;
  if (video.getAttribute("src") !== url) video.src = url;
  video.hidden = false;
  empty.hidden = true;
  button.disabled = false;
  button.onclick = () => {
    video.scrollIntoView({ behavior: "smooth", block: "center" });
    video.play().catch(() => {});
  };
  badge.textContent = "SAVED CLIP";
  timestamp.textContent = formatDate(recording.timestamp, { dateStyle: "medium", timeStyle: "short" });
  timestamp.hidden = false;
  byId("latestCamera").textContent = recording.camera || "CAMERA-01";
  byId("latestAlertStatus").textContent = recording.status || "CRITICAL";
  byId("latestAlertStatus").classList.remove("text-muted");
}

function renderFeaturedCandidate(event) {
  const video = byId("candidateFeaturedVideo");
  hasFeaturedCandidate = Boolean(event?.clip_url);
  if (!hasFeaturedCandidate) {
    video.pause();
    video.removeAttribute("src");
    video.load();
    updateFeaturedClipsVisibility();
    return;
  }
  const url = `${API_BASE}${event.clip_url}`;
  if (video.getAttribute("src") !== url) video.src = url;
  byId("candidateFeaturedName").textContent = event.clip_filename || "Candidate video";
  byId("candidateFeaturedTime").textContent = formatDate(event.timestamp, { dateStyle: "medium", timeStyle: "short" });
  const score = event.decision_score ?? event.clip_score;
  byId("candidateFeaturedScore").textContent = Number.isFinite(Number(score)) ? `${(Number(score) * 100).toFixed(1)}% · pending review` : "Pending human review";
  updateFeaturedClipsVisibility();
}

function renderEmergencyRecordings(events) {
  const clips = [...events].reverse().filter((event) => event.video);
  byId("recordingCount").textContent = String(clips.length);
  byId("navRecordingCount").textContent = String(clips.length);
  const grid = byId("recordingGrid");
  grid.replaceChildren();
  if (!clips.length) {
    grid.appendChild(makeEmpty("No saved emergency clips are linked to events yet."));
    return;
  }
  clips.slice(0, 1).forEach((event) => {
    grid.appendChild(makeClipCard({
      url: `${API_BASE}/videos/${encodeURIComponent(event.video)}`,
      kind: "EMERGENCY",
      title: event.video,
      time: event.timestamp,
      meta: event.camera || "CAMERA-01",
    }));
  });
}

function renderCandidates(events) {
  const grid = byId("candidateGrid");
  grid.replaceChildren();
  const candidates = Array.isArray(events) ? events : [];
  byId("candidateCount").textContent = String(candidates.length);
  byId("navCandidateCount").textContent = String(candidates.length);
  byId("candidateReviewBadge").textContent = `${candidates.length} TO REVIEW`;
  if (!candidates.length) {
    renderFeaturedCandidate(null);
    grid.appendChild(makeEmpty("No Phase 2 candidate clips have been recorded yet."));
    return;
  }

  renderFeaturedCandidate(candidates.find((event) => event.clip_url) || null);

  candidates.slice(0, 1).forEach((event) => {
    if (!event.clip_url) {
      const missing = makeEmpty(`Candidate clip unavailable: ${event.clip_filename || "video file not found"}`);
      grid.appendChild(missing);
      return;
    }
    const filename = event.clip_filename || "";
    const clipUrl = `${API_BASE}${event.clip_url}`;
    const originalUrl = `${API_BASE}/candidate-clips/${encodeURIComponent(filename)}`;
    const card = makeClipCard({
      url: clipUrl,
      kind: "PHASE 2 CANDIDATE",
      title: event.clip_filename || "Candidate video",
      time: event.timestamp,
      meta: `${event.consecutive_high_clips ?? "—"} high clips`,
      score: event.decision_score ?? event.clip_score,
      candidate: true,
    });
    const video = card.querySelector("video");
    video.dataset.originalUrl = originalUrl;
    video.addEventListener("error", () => {
      if (video.dataset.usingOriginal !== "true") {
        video.dataset.usingOriginal = "true";
        video.src = originalUrl;
      } else {
        video.hidden = true;
        const message = document.createElement("p");
        message.className = "video-unavailable";
        message.textContent = "This browser cannot play the clip. Download the original video below.";
        video.parentElement.insertBefore(message, video.nextSibling);
      }
    });
    const download = document.createElement("a");
    download.className = "clip-download";
    download.href = originalUrl;
    download.download = filename;
    download.textContent = "Download original clip ↗";
    card.querySelector(".clip-card-body").appendChild(download);
    grid.appendChild(card);
  });
}

function renderActivity(events) {
  const list = byId("activityList");
  list.replaceChildren();
  if (!events.length) {
    list.appendChild(makeEmpty("No emergency events have been logged."));
    return;
  }
  [...events].reverse().slice(0, 8).forEach((event) => {
    const row = document.createElement("div");
    row.className = "activity-row";
    const symbol = document.createElement("span");
    symbol.className = "activity-symbol";
    symbol.textContent = "!";
    const copy = document.createElement("div");
    copy.className = "activity-copy";
    const title = document.createElement("strong");
    title.textContent = event.event === "EMERGENCY_DETECTED" ? "Emergency gesture detected" : (event.event || "Emergency event");
    const detail = document.createElement("span");
    detail.textContent = `${event.camera || "CAMERA-01"} · ${event.status || "Recorded"}`;
    copy.append(title, detail);
    const time = document.createElement("time");
    time.className = "activity-time";
    time.dateTime = event.timestamp || "";
    time.textContent = formatDate(event.timestamp, { dateStyle: "medium", timeStyle: "short" });
    const action = document.createElement("button");
    action.className = "activity-action";
    action.type = "button";
    action.textContent = event.video ? "Play clip ↗" : "No clip";
    action.disabled = !event.video;
    if (event.video) action.addEventListener("click", () => {
      renderFeaturedRecording(event);
      location.hash = "recordings";
      byId("featuredVideo").play().catch(() => {});
    });
    row.append(symbol, copy, time, action);
    list.appendChild(row);
  });
}

function renderLatestAlert(events) {
  const latest = events.length ? events[events.length - 1] : null;
  const icon = byId("alertSummaryIcon");
  if (!latest) {
    icon.textContent = "✓";
    icon.classList.remove("danger");
    byId("alertSummaryTitle").textContent = "All clear";
    byId("alertSummaryText").textContent = "No emergency events recorded.";
    byId("alertSummaryCamera").textContent = "CAMERA-01";
    byId("alertSummaryTime").textContent = "—";
    byId("latestAlertStatus").textContent = "No event selected";
    return;
  }
  icon.textContent = "!";
  icon.classList.add("danger");
  byId("alertSummaryTitle").textContent = "Emergency detected";
  byId("alertSummaryText").textContent = latest.video ? "Recording saved for review." : "Event logged without an attached video.";
  byId("alertSummaryCamera").textContent = latest.camera || "CAMERA-01";
  byId("alertSummaryTime").textContent = formatDate(latest.timestamp, { dateStyle: "medium", timeStyle: "short" });
}

function renderPhase2Status(status) {
  const state = String(status?.state || "UNKNOWN").toUpperCase();
  byId("workerState").textContent = state.toLowerCase();
  byId("workerMessage").textContent = status?.message || "Waiting for analysis";
  byId("phase2State").textContent = state;
  const stateClass = state === "ERROR" || state === "OFFLINE" || state === "SAVE_ERROR"
    ? "error" : state === "CANDIDATE" ? "candidate" : state === "ONLINE" || state === "MONITORING" ? "online" : "";
  byId("phase2State").className = `detector-status ${stateClass}`;
  byId("workerDot").className = `metric-dot ${stateClass === "error" ? "red-dot" : stateClass === "candidate" ? "amber-dot" : stateClass === "online" ? "blue-dot" : ""}`;
  const score = Number(status?.decision_score ?? status?.clip_score);
  const validScore = Number.isFinite(score);
  byId("decisionScore").textContent = validScore ? `${(score * 100).toFixed(1)}%` : "—";
  byId("scoreBar").style.width = validScore ? `${Math.max(0, Math.min(100, score * 100))}%` : "0";
  const threshold = Number(status?.threshold);
  byId("scoreThreshold").textContent = Number.isFinite(threshold) ? `Candidate threshold ${(threshold * 100).toFixed(0)}%` : "Candidate threshold —";
  byId("scoreUpdated").textContent = status?.updated_at ? formatDate(status.updated_at, { timeStyle: "short" }) : "No analysis yet";
}

function showEmergencyModal(event) {
  const modal = byId("emergencyModal");
  const video = byId("modalVideo");
  byId("modalMessage").textContent = `${event.camera || "Camera"} · ${formatDate(event.timestamp, { dateStyle: "medium", timeStyle: "short" })}`;
  if (event.video) {
    video.src = `${API_BASE}/videos/${encodeURIComponent(event.video)}`;
    video.hidden = false;
    byId("modalNoVideo").hidden = true;
    video.play().catch(() => {});
  } else {
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.hidden = true;
    byId("modalNoVideo").hidden = false;
  }
  modal.hidden = false;
  byId("closeModal").focus();
}

function closeEmergencyModal() {
  byId("modalVideo").pause();
  byId("emergencyModal").hidden = true;
}

async function refreshDashboard() {
  if (refreshInProgress) return;
  refreshInProgress = true;
  try {
    const [healthResult, eventsResult, candidatesResult, statusResult] = await Promise.allSettled([
      fetchJSON("/health"), fetchJSON("/events"), fetchJSON("/phase2-events"), fetchJSON("/phase2-status"),
    ]);
    const backendOnline = healthResult.status === "fulfilled" && healthResult.value.status === "ok";
    setBackendState(backendOnline);
    if (healthResult.status === "fulfilled") {
      byId("emergencyCount").textContent = String(healthResult.value.phase1_events ?? 0);
    }

    if (eventsResult.status === "fulfilled" && Array.isArray(eventsResult.value)) {
      const events = eventsResult.value;
      byId("emergencyCount").textContent = String(events.length);
      const key = events.length ? eventKey(events[events.length - 1]) : null;
      if (hasLoadedEvents && key && key !== lastEventKey) showEmergencyModal(events[events.length - 1]);
      lastEventKey = key;
      hasLoadedEvents = true;
      renderLatestAlert(events);
      const emergencySnapshot = JSON.stringify(events.map(({ id, timestamp, video, status, camera }) => [id, timestamp, video, status, camera]));
      if (emergencySnapshot !== previousEmergencySnapshot) {
        renderEmergencyRecordings(events);
        renderActivity(events);
        previousEmergencySnapshot = emergencySnapshot;
      }
      renderFeaturedRecording([...events].reverse().find((event) => event.video) || null);
    } else {
      setBackendState(false);
      byId("activityList").replaceChildren(makeEmpty("Event history is unavailable until the backend responds."));
      byId("recordingGrid").replaceChildren(makeEmpty("Emergency recordings are unavailable until the backend responds."));
      byId("connectionMessage").textContent = "Could not load emergency events. Start the FastAPI backend and refresh.";
    }

    if (candidatesResult.status === "fulfilled" && Array.isArray(candidatesResult.value)) {
      const candidateSnapshot = JSON.stringify(candidatesResult.value);
      if (candidateSnapshot !== previousCandidateSnapshot) {
        renderCandidates(candidatesResult.value);
        previousCandidateSnapshot = candidateSnapshot;
      }
    } else {
      byId("candidateGrid").replaceChildren(makeEmpty("Phase 2 candidate events are unavailable."));
    }
    if (statusResult.status === "fulfilled") renderPhase2Status(statusResult.value);
    else {
      byId("workerState").textContent = "unknown";
      byId("workerMessage").textContent = "Phase 2 status unavailable";
      byId("phase2State").textContent = "UNKNOWN";
    }
    byId("lastUpdated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    console.error("Dashboard refresh failed", error);
    setBackendState(false);
  } finally {
    refreshInProgress = false;
  }
}

function updateClock() {
  byId("currentTime").textContent = new Date().toLocaleString(undefined, {
    weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

byId("refreshButton").addEventListener("click", refreshDashboard);
byId("closeModal").addEventListener("click", closeEmergencyModal);
byId("emergencyModal").addEventListener("click", (event) => {
  if (event.target === byId("emergencyModal")) closeEmergencyModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !byId("emergencyModal").hidden) closeEmergencyModal();
});

const navLinks = [...document.querySelectorAll(".nav-link")];
navLinks.forEach((link) => link.addEventListener("click", () => {
  navLinks.forEach((item) => item.classList.toggle("active", item === link));
}));

updateClock();
renderCameraGrid();
refreshDashboard();
setInterval(updateClock, 30_000);
setInterval(refreshDashboard, 5_000);
