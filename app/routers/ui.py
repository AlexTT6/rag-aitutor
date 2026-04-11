from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RAG Tutor — Test Panel</title>
<style>
  :root {
    --bg: #0f1117;
    --card: #1a1d27;
    --border: #2a2d3e;
    --accent: #6c63ff;
    --accent2: #00d4aa;
    --text: #e2e8f0;
    --muted: #8892a4;
    --success: #22c55e;
    --warning: #f59e0b;
    --error: #ef4444;
    --radius: 12px;
    --font: 'Inter', system-ui, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); min-height: 100vh; }

  header {
    background: var(--card);
    border-bottom: 1px solid var(--border);
    padding: 20px 40px;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  header .logo { font-size: 22px; font-weight: 700; color: var(--accent); letter-spacing: -0.5px; }
  header .subtitle { font-size: 13px; color: var(--muted); }
  header .status-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--muted);
    margin-left: auto; transition: background 0.3s;
  }
  header .status-dot.ok { background: var(--success); box-shadow: 0 0 8px var(--success); }
  header .status-text { font-size: 12px; color: var(--muted); }

  main { max-width: 900px; margin: 40px auto; padding: 0 20px 60px; }

  .steps { display: flex; flex-direction: column; gap: 24px; }

  .step {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .step.active { border-color: var(--accent); }
  .step.done { border-color: var(--accent2); }

  .step-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 24px;
    cursor: pointer;
    user-select: none;
  }
  .step-num {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700; color: var(--muted);
    flex-shrink: 0; transition: all 0.2s;
  }
  .step.active .step-num { background: var(--accent); color: #fff; }
  .step.done .step-num { background: var(--accent2); color: #fff; }
  .step-title { font-size: 16px; font-weight: 600; }
  .step-desc { font-size: 13px; color: var(--muted); margin-top: 2px; }
  .step-badge {
    margin-left: auto;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    background: var(--border);
    color: var(--muted);
  }
  .step.done .step-badge { background: rgba(0,212,170,0.15); color: var(--accent2); }

  .step-body { padding: 0 24px 24px; display: none; }
  .step.active .step-body, .step.done .step-body { display: block; }

  label { display: block; font-size: 12px; font-weight: 600; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }

  input[type=text], input[type=number], textarea {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 14px;
    padding: 10px 14px;
    outline: none;
    transition: border-color 0.2s;
    font-family: var(--font);
  }
  input:focus, textarea:focus { border-color: var(--accent); }
  textarea { resize: vertical; min-height: 80px; }

  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .field { margin-bottom: 16px; }

  .file-drop {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 32px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    position: relative;
    overflow: hidden;
  }
  .file-drop:hover, .file-drop.dragover { border-color: var(--accent); background: rgba(108,99,255,0.05); }
  .file-drop input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .file-drop .icon { font-size: 32px; margin-bottom: 10px; }
  .file-drop .label { font-size: 14px; color: var(--muted); }
  .file-drop .label strong { color: var(--accent); }
  .file-drop .selected { font-size: 13px; color: var(--accent2); margin-top: 8px; font-weight: 600; }

  btn, .btn {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 14px; font-weight: 600;
    padding: 10px 20px;
    cursor: pointer;
    transition: all 0.2s;
    font-family: var(--font);
  }
  .btn:hover { opacity: 0.85; transform: translateY(-1px); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn.secondary { background: var(--border); color: var(--text); }
  .btn.green { background: var(--accent2); color: #111; }
  .btn-row { display: flex; gap: 10px; align-items: center; }

  .info-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 13px;
    margin-top: 12px;
  }
  .info-box .row-item { display: flex; justify-content: space-between; padding: 4px 0; }
  .info-box .key { color: var(--muted); }
  .info-box .val { font-family: 'SF Mono', monospace; color: var(--text); font-size: 12px; }
  .info-box .val.copy { cursor: pointer; color: var(--accent); }
  .info-box .val.copy:hover { text-decoration: underline; }

  .status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .status-pill.uploaded { background: rgba(245,158,11,0.15); color: var(--warning); }
  .status-pill.processing { background: rgba(108,99,255,0.15); color: var(--accent); }
  .status-pill.indexed { background: rgba(34,197,94,0.15); color: var(--success); }
  .status-pill.failed { background: rgba(239,68,68,0.15); color: var(--error); }

  .progress-bar {
    height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 12px;
  }
  .progress-bar .fill {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    animation: pulse 1.5s ease-in-out infinite;
    width: 40%;
  }
  @keyframes pulse {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(350%); }
  }
  .progress-bar.done .fill { animation: none; width: 100%; background: var(--success); }
  .progress-bar.failed .fill { animation: none; width: 100%; background: var(--error); }

  .results { margin-top: 16px; display: flex; flex-direction: column; gap: 12px; }
  .result-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    transition: border-color 0.2s;
  }
  .result-card:hover { border-color: var(--accent); }
  .result-card.low { border-color: var(--warning); }
  .result-meta {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px;
  }
  .result-meta .filename { font-size: 13px; font-weight: 600; color: var(--accent2); }
  .result-meta .page { font-size: 11px; color: var(--muted); background: var(--border); padding: 2px 8px; border-radius: 4px; }
  .result-meta .score {
    margin-left: auto;
    font-size: 13px; font-weight: 700;
  }
  .score.high { color: var(--success); }
  .score.med { color: var(--warning); }
  .score.low { color: var(--error); }
  .result-text {
    font-size: 13px; color: var(--muted); line-height: 1.7;
    max-height: 120px; overflow: hidden;
    position: relative;
  }
  .result-text.expanded { max-height: none; }
  .expand-btn { font-size: 12px; color: var(--accent); cursor: pointer; margin-top: 6px; display: inline-block; }

  .alert {
    padding: 12px 16px; border-radius: 8px;
    font-size: 13px; margin-top: 12px;
    display: flex; align-items: flex-start; gap: 10px;
  }
  .alert.error { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); color: var(--error); }
  .alert.success { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); color: var(--success); }
  .alert.info { background: rgba(108,99,255,0.1); border: 1px solid rgba(108,99,255,0.3); color: #a5b4fc; }

  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid currentColor; border-top-color: transparent; border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .sep { border: none; border-top: 1px solid var(--border); margin: 20px 0; }
  .empty { text-align: center; padding: 40px; color: var(--muted); font-size: 14px; }

  .course-id-display {
    font-family: 'SF Mono', monospace;
    font-size: 12px;
    color: var(--accent);
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.3);
    padding: 8px 14px;
    border-radius: 6px;
    cursor: pointer;
    display: inline-flex; align-items: center; gap: 8px;
  }
  .course-id-display:hover { background: rgba(108,99,255,0.2); }

  .log {
    font-family: 'SF Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    max-height: 120px;
    overflow-y: auto;
    margin-top: 10px;
    white-space: pre-wrap;
  }
  .log .ok { color: var(--success); }
  .log .err { color: var(--error); }
  .log .info { color: var(--accent); }
</style>
</head>
<body>

<header>
  <div>
    <div class="logo">RAG Tutor</div>
    <div class="subtitle">Test Panel — End-to-end verification</div>
  </div>
  <div class="status-text" id="healthText">Checking server...</div>
  <div class="status-dot" id="healthDot"></div>
</header>

<main>
  <div class="steps">

    <!-- STEP 1: Create Course -->
    <div class="step active" id="step1">
      <div class="step-header" onclick="toggleStep('step1')">
        <div class="step-num">1</div>
        <div>
          <div class="step-title">Create Course</div>
          <div class="step-desc">Register a course to group your documents</div>
        </div>
        <div class="step-badge" id="badge1">Required</div>
      </div>
      <div class="step-body">
        <div class="field">
          <label>Course Name</label>
          <input type="text" id="courseName" placeholder="e.g. Introduction to Machine Learning" value="Test Course"/>
        </div>
        <div class="btn-row">
          <button class="btn" onclick="createCourse()">Create Course</button>
        </div>
        <div id="courseResult"></div>
      </div>
    </div>

    <!-- STEP 2: Upload File -->
    <div class="step" id="step2">
      <div class="step-header" onclick="toggleStep('step2')">
        <div class="step-num">2</div>
        <div>
          <div class="step-title">Upload PDF</div>
          <div class="step-desc">Upload a PDF document for ingestion into RAG</div>
        </div>
        <div class="step-badge" id="badge2">Waiting</div>
      </div>
      <div class="step-body">
        <div class="field">
          <label>Course ID</label>
          <input type="text" id="uploadCourseId" placeholder="Auto-filled from Step 1"/>
        </div>
        <div class="field">
          <div class="file-drop" id="fileDrop">
            <input type="file" accept=".pdf" id="fileInput" onchange="onFileSelected(event)"/>
            <div class="icon">📄</div>
            <div class="label">Drag & drop a <strong>PDF</strong> here, or click to browse</div>
            <div class="selected" id="fileLabel"></div>
          </div>
        </div>
        <div class="btn-row">
          <button class="btn" onclick="uploadFile()" id="uploadBtn">Upload & Ingest</button>
        </div>
        <div id="uploadResult"></div>
      </div>
    </div>

    <!-- STEP 3: Poll Status -->
    <div class="step" id="step3">
      <div class="step-header" onclick="toggleStep('step3')">
        <div class="step-num">3</div>
        <div>
          <div class="step-title">Ingestion Status</div>
          <div class="step-desc">Monitor the pipeline: extract → chunk → embed → index</div>
        </div>
        <div class="step-badge" id="badge3">Waiting</div>
      </div>
      <div class="step-body">
        <div class="field">
          <label>File ID</label>
          <input type="text" id="pollFileId" placeholder="Auto-filled after upload"/>
        </div>
        <div class="btn-row">
          <button class="btn secondary" onclick="pollOnce()">Check Now</button>
          <button class="btn" onclick="startPolling()" id="autoBtn">Auto-poll (3s)</button>
          <button class="btn secondary" onclick="stopPolling()" id="stopBtn" style="display:none">Stop</button>
        </div>
        <div id="pollResult"></div>
        <div class="log" id="pollLog" style="display:none"></div>
      </div>
    </div>

    <!-- STEP 4: Retrieve / Search -->
    <div class="step" id="step4">
      <div class="step-header" onclick="toggleStep('step4')">
        <div class="step-num">4</div>
        <div>
          <div class="step-title">Test Retrieval</div>
          <div class="step-desc">Run a semantic search query against indexed documents</div>
        </div>
        <div class="step-badge" id="badge4">Waiting</div>
      </div>
      <div class="step-body">
        <div class="row">
          <div class="field">
            <label>Course ID</label>
            <input type="text" id="retrieveCourseId" placeholder="Auto-filled from Step 1"/>
          </div>
          <div class="field">
            <label>Top K results</label>
            <input type="number" id="topK" value="5" min="1" max="20"/>
          </div>
        </div>
        <div class="field">
          <label>Search Query</label>
          <textarea id="query" placeholder="What is backpropagation? How does gradient descent work? Explain neural networks..."></textarea>
        </div>
        <div class="btn-row">
          <button class="btn green" onclick="doRetrieve()" id="retrieveBtn">Search</button>
        </div>
        <div id="retrieveResult"></div>
      </div>
    </div>

  </div>
</main>

<script>
const API = '';  // same origin
let courseId = null;
let fileId = null;
let pollTimer = null;
let pollLogs = [];

// ─── Health Check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(API + '/health');
    const ok = r.ok;
    document.getElementById('healthDot').className = 'status-dot' + (ok ? ' ok' : '');
    document.getElementById('healthText').textContent = ok ? 'Server online' : 'Server error';
  } catch {
    document.getElementById('healthDot').className = 'status-dot';
    document.getElementById('healthText').textContent = 'Server offline';
  }
}
checkHealth();
setInterval(checkHealth, 10000);

// ─── Toggle Step ──────────────────────────────────────────────────────────────
function toggleStep(id) {
  const el = document.getElementById(id);
  const isActive = el.classList.contains('active');
  // Only collapse, don't activate locked steps
}

function activateStep(id) {
  document.getElementById(id).classList.add('active');
}

// ─── Step 1: Create Course ────────────────────────────────────────────────────
async function createCourse() {
  const name = document.getElementById('courseName').value.trim();
  if (!name) return showAlert('courseResult', 'error', 'Enter a course name.');

  const btn = event.target;
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Creating...';

  try {
    const r = await fetch(API + '/courses', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || JSON.stringify(data));

    courseId = data.course_id;
    document.getElementById('uploadCourseId').value = courseId;
    document.getElementById('retrieveCourseId').value = courseId;

    document.getElementById('courseResult').innerHTML = `
      <div class="alert success">✓ Course created successfully</div>
      <div class="info-box" style="margin-top:10px">
        <div class="row-item"><span class="key">Course ID</span><span class="val copy" onclick="copyText('${courseId}')" title="Click to copy">${courseId}</span></div>
        <div class="row-item"><span class="key">Name</span><span class="val">${escHtml(data.name)}</span></div>
        <div class="row-item"><span class="key">Created</span><span class="val">${new Date(data.created_at).toLocaleString()}</span></div>
      </div>`;

    document.getElementById('step1').classList.add('done');
    document.getElementById('badge1').textContent = '✓ Done';
    activateStep('step2');
    document.getElementById('step2').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (e) {
    showAlert('courseResult', 'error', e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Create Course';
  }
}

// ─── Step 2: Upload File ──────────────────────────────────────────────────────
function onFileSelected(e) {
  const f = e.target.files[0];
  document.getElementById('fileLabel').textContent = f ? `Selected: ${f.name} (${(f.size/1024/1024).toFixed(2)} MB)` : '';
}

// Drag & drop styling
const drop = document.getElementById('fileDrop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('dragover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f && f.type === 'application/pdf') {
    document.getElementById('fileInput').files = e.dataTransfer.files;
    document.getElementById('fileLabel').textContent = `Selected: ${f.name} (${(f.size/1024/1024).toFixed(2)} MB)`;
  }
});

async function uploadFile() {
  const cid = document.getElementById('uploadCourseId').value.trim();
  const fi = document.getElementById('fileInput').files[0];
  if (!cid) return showAlert('uploadResult', 'error', 'Enter a Course ID.');
  if (!fi) return showAlert('uploadResult', 'error', 'Select a PDF file.');

  const btn = document.getElementById('uploadBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Uploading...';

  const fd = new FormData();
  fd.append('file', fi);
  fd.append('course_id', cid);

  try {
    const r = await fetch(API + '/files/upload', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || JSON.stringify(data));

    fileId = data.file_id;
    document.getElementById('pollFileId').value = fileId;

    document.getElementById('uploadResult').innerHTML = `
      <div class="alert success">✓ File uploaded — ingestion started in background</div>
      <div class="info-box" style="margin-top:10px">
        <div class="row-item"><span class="key">File ID</span><span class="val copy" onclick="copyText('${fileId}')" title="Click to copy">${fileId}</span></div>
        <div class="row-item"><span class="key">Filename</span><span class="val">${escHtml(data.filename)}</span></div>
        <div class="row-item"><span class="key">Status</span><span class="val">${data.status}</span></div>
      </div>`;

    document.getElementById('step2').classList.add('done');
    document.getElementById('badge2').textContent = '✓ Done';
    activateStep('step3');
    document.getElementById('step3').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Auto-start polling
    setTimeout(() => startPolling(), 500);
  } catch (e) {
    showAlert('uploadResult', 'error', e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Upload & Ingest';
  }
}

// ─── Step 3: Status Polling ───────────────────────────────────────────────────
function startPolling() {
  if (pollTimer) return;
  document.getElementById('autoBtn').style.display = 'none';
  document.getElementById('stopBtn').style.display = '';
  document.getElementById('pollLog').style.display = '';
  pollTimer = setInterval(pollOnce, 3000);
  pollOnce();
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = null;
  document.getElementById('autoBtn').style.display = '';
  document.getElementById('stopBtn').style.display = 'none';
}

function addLog(msg, cls='') {
  const ts = new Date().toLocaleTimeString();
  pollLogs.push(`<span class="${cls}">[${ts}] ${msg}</span>`);
  if (pollLogs.length > 30) pollLogs.shift();
  const el = document.getElementById('pollLog');
  el.innerHTML = pollLogs.join('\\n');
  el.scrollTop = el.scrollHeight;
}

async function pollOnce() {
  const fid = document.getElementById('pollFileId').value.trim();
  if (!fid) return showAlert('pollResult', 'error', 'Enter a File ID.');

  try {
    const r = await fetch(API + '/files/' + fid);
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || JSON.stringify(data));

    renderPollResult(data);

    if (data.status === 'indexed') {
      addLog(`indexed — ${data.chunk_count} chunks`, 'ok');
      stopPolling();
      document.getElementById('step3').classList.add('done');
      document.getElementById('badge3').textContent = `✓ ${data.chunk_count} chunks`;
      activateStep('step4');
      document.getElementById('step4').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (data.status === 'failed') {
      addLog('failed: ' + (data.error_message || 'unknown error'), 'err');
      stopPolling();
      document.getElementById('badge3').textContent = 'Failed';
    } else {
      addLog(`status: ${data.status}`, 'info');
    }
  } catch (e) {
    addLog('error: ' + e.message, 'err');
  }
}

function renderPollResult(data) {
  const statusClass = data.status;
  const progressClass = data.status === 'indexed' ? 'done' : data.status === 'failed' ? 'failed' : '';
  document.getElementById('pollResult').innerHTML = `
    <div class="info-box" style="margin-top:12px">
      <div class="row-item"><span class="key">Status</span><span class="val"><span class="status-pill ${statusClass}">${data.status}</span></span></div>
      ${data.chunk_count != null ? `<div class="row-item"><span class="key">Chunks indexed</span><span class="val" style="color:var(--success)">${data.chunk_count}</span></div>` : ''}
      ${data.error_message ? `<div class="row-item"><span class="key">Error</span><span class="val" style="color:var(--error)">${escHtml(data.error_message)}</span></div>` : ''}
      ${data.indexed_at ? `<div class="row-item"><span class="key">Indexed at</span><span class="val">${new Date(data.indexed_at).toLocaleString()}</span></div>` : ''}
    </div>
    <div class="progress-bar ${progressClass}"><div class="fill"></div></div>`;
}

// ─── Step 4: Retrieve ─────────────────────────────────────────────────────────
async function doRetrieve() {
  const cid = document.getElementById('retrieveCourseId').value.trim();
  const q = document.getElementById('query').value.trim();
  const topK = parseInt(document.getElementById('topK').value) || 5;

  if (!cid) return showAlert('retrieveResult', 'error', 'Enter a Course ID.');
  if (!q) return showAlert('retrieveResult', 'error', 'Enter a search query.');

  const btn = document.getElementById('retrieveBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Searching...';

  try {
    const r = await fetch(API + '/retrieve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, course_id: cid, top_k: topK })
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || JSON.stringify(data));

    renderResults(data.results);
    document.getElementById('badge4').textContent = `${data.results.length} results`;
    document.getElementById('step4').classList.add('done');
  } catch (e) {
    showAlert('retrieveResult', 'error', e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Search';
  }
}

function renderResults(results) {
  if (!results.length) {
    document.getElementById('retrieveResult').innerHTML = `<div class="empty">No results found. Try a different query or check that the file is indexed.</div>`;
    return;
  }

  const cards = results.map((r, i) => {
    const sc = r.score;
    const scoreClass = sc >= 0.7 ? 'high' : sc >= 0.5 ? 'med' : 'low';
    const cardClass = r.low_confidence ? 'low' : '';
    const textId = 'rt_' + i;
    return `
      <div class="result-card ${cardClass}">
        <div class="result-meta">
          <span class="filename">📄 ${escHtml(r.filename)}</span>
          <span class="page">Page ${r.page}</span>
          ${r.low_confidence ? '<span class="status-pill" style="background:rgba(245,158,11,0.15);color:var(--warning)">Low confidence</span>' : ''}
          <span class="score ${scoreClass}">${(sc * 100).toFixed(1)}%</span>
        </div>
        <div class="result-text" id="${textId}">${escHtml(r.text)}</div>
        <span class="expand-btn" onclick="toggleExpand('${textId}', this)">Show more ↓</span>
      </div>`;
  }).join('');

  document.getElementById('retrieveResult').innerHTML = `
    <div class="alert info" style="margin-bottom:8px">Found ${results.length} chunk(s) — scores show semantic similarity to your query</div>
    <div class="results">${cards}</div>`;
}

function toggleExpand(id, btn) {
  const el = document.getElementById(id);
  if (el.classList.contains('expanded')) {
    el.classList.remove('expanded'); btn.textContent = 'Show more ↓';
  } else {
    el.classList.add('expanded'); btn.textContent = 'Show less ↑';
  }
}

// ─── Utils ────────────────────────────────────────────────────────────────────
function showAlert(containerId, type, msg) {
  const icon = type === 'error' ? '✕' : type === 'success' ? '✓' : 'ℹ';
  document.getElementById(containerId).innerHTML = `<div class="alert ${type}">${icon} ${escHtml(msg)}</div>`;
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function copyText(t) {
  navigator.clipboard.writeText(t).then(() => {
    // brief visual feedback
    event.target.textContent = 'Copied!';
    setTimeout(() => { event.target.textContent = t; }, 1200);
  });
}
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def test_ui() -> str:
    return _HTML
