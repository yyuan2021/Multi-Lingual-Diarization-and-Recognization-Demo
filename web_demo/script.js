/* ── Element refs ──────────────────────────────────────────── */
const recordBtn      = document.getElementById("recordBtn");
const fileInput      = document.getElementById("fileInput");
const sampleBtn      = document.getElementById("sampleBtn");
const langSelect     = document.getElementById("langSelect");
const runBtn         = document.getElementById("runBtn");
const recordStatus   = document.getElementById("recordStatus");
const timerEl        = document.getElementById("timer");
const originalWrap   = document.getElementById("originalWrap");
const originalPlayer = document.getElementById("originalPlayer");
const progressWrap   = document.getElementById("progressWrap");
const resultsSection = document.getElementById("resultsSection");

/* ── State ─────────────────────────────────────────────────── */
let audioBlob     = null;
let mediaRecorder = null;
let chunks        = [];
let timerInterval = null;
let elapsed       = 0;
let isRecording   = false;
let isSampleMode = false;
let currentJobId = null;
let stopPolling  = false;

/* ── Helpers ───────────────────────────────────────────────── */
function setAudioBlob(blob, filename) {
  audioBlob = blob instanceof File ? blob : new File([blob], filename || "audio.wav");
  originalPlayer.src = URL.createObjectURL(blob);
  originalWrap.classList.remove("hidden");
  runBtn.disabled = false;
}

function setField(id, text) {
  document.getElementById(id).textContent = text || "—";
}

/* ── File upload ───────────────────────────────────────────── */
fileInput.addEventListener("change", e => {
  const file = e.target.files[0];
  if (file) setAudioBlob(file, file.name);
});

/* ── Load sample ───────────────────────────────────────────── */
sampleBtn.addEventListener("click", async () => {
  isSampleMode = true;
  resultsSection.classList.add("hidden");
  
  const targetLang = langSelect.value;
  sampleBtn.disabled = true;
  sampleBtn.textContent = "Loading…";

  resultsSection.classList.add("hidden");

  try {
    const res = await fetch(`/sample?target_lang=${targetLang}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    window.sampleData = data;

    runBtn.disabled = false;

  } catch (err) {
    alert("Could not load sample: " + err.message);
  } finally {
    sampleBtn.disabled = false;
    sampleBtn.textContent = "♫ Load Sample";
  }
});

/* ── Recording ─────────────────────────────────────────────── */
recordBtn.addEventListener("click", async () => {
  isRecording ? stopRecording() : await startRecording();
});

async function webmToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  const wavBuffer = encodeWAV(audioBuffer);
  return new Blob([wavBuffer], { type: "audio/wav" });
}

function encodeWAV(audioBuffer) {
  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const length = audioBuffer.length * numChannels * 2;
  const buffer = new ArrayBuffer(44 + length);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + length, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * 2, true);
  view.setUint16(32, numChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, length, true);

  let offset = 44;
  for (let i = 0; i < audioBuffer.length; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      let sample = audioBuffer.getChannelData(ch)[i];
      sample = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, sample * 0x7fff, true);
      offset += 2;
    }
  }

  return buffer;
}


async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    chunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) {
        chunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const webmBlob = new Blob(chunks, { type: "audio/webm" });

      console.log("WEBM:", webmBlob.type, webmBlob.size);

      const wavBlob = await webmToWav(webmBlob);

      console.log("WAV:", wavBlob.type, wavBlob.size);

      setAudioBlob(wavBlob, "recording.wav");

      stream.getTracks().forEach(t => t.stop());
    };

    mediaRecorder.start();
    isRecording = true;

    elapsed = 0;
    timerEl.textContent = "0s";
    timerInterval = setInterval(() => {
      timerEl.textContent = `${++elapsed}s`;
    }, 1000);

    recordBtn.textContent = "⏹ Stop Recording";
    recordBtn.classList.add("active");
    recordStatus.classList.remove("hidden");

  } catch (err) {
    alert("Microphone error: " + err.message);
  }
}


function stopRecording() {
  mediaRecorder.stop();
  isRecording = false;
  clearInterval(timerInterval);
  recordBtn.textContent = "⏺ Start Recording";
  recordBtn.classList.remove("active");
  recordStatus.classList.add("hidden");
}

/* ── Run pipeline ──────────────────────────────────────────── */
runBtn.addEventListener("click", async () => {
  isSampleMode = false;
  progressWrap.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  runBtn.disabled = true;

  try {
    if (window.sampleData) {
      await new Promise(r => setTimeout(r, 1000));
      resultsSection.classList.remove("hidden");
      displayResults(window.sampleData);
      window.sampleData = null;
      progressWrap.classList.add("hidden");
      return;
    }

    if (!audioBlob) return;

    const form = new FormData();
    form.append("audio", audioBlob, audioBlob.name || "audio.wav");
    form.append("target_lang", langSelect.value);

    const res = await fetch("/process", { method: "POST", body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `HTTP ${res.status}`);
    }

    const { job_id } = await res.json();
    currentJobId = job_id;
    stopPolling  = false;
    await pollUntilDone(job_id);

  } catch (err) {
    progressWrap.classList.add("hidden");
    alert("Pipeline error: " + err.message);
  } finally {
    runBtn.disabled = false;
  }
});

/* ── Job polling ───────────────────────────────────────────── */
async function pollUntilDone(jobId) {
  while (!stopPolling) {
    const res = await fetch(`/progress/${jobId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (jobId !== currentJobId) return;

    updateStepUI(data.step, data.message);

    if (data.done) {
      progressWrap.classList.add("hidden");
      if (data.error) throw new Error(data.error);
      displayResults(data.result);
      break;
    }

    await new Promise(r => setTimeout(r, 800));
  }
}

function updateStepUI(_step, message) {
  const p = progressWrap.querySelector("p");
  if (p) p.textContent = message || "Processing…";
}

/* ── Display results ───────────────────────────────────────── */
function displayResults(data) {
  if (isSampleMode) return;
  document.getElementById("spk0Player").src = data.spk0_audio;
  document.getElementById("spk1Player").src = data.spk1_audio;
  setField("spk0Text",  data.spk0_text);
  setField("spk1Text",  data.spk1_text);
  setField("spk0Trans", data.spk0_en);
  setField("spk1Trans", data.spk1_en);

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}
