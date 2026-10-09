// SpectroMood dashboard: captures the mic, posts rolling windows to /api/analyze,
// and renders Clef's answers. Several windows can be in flight at once, so every result
// is placed on the timeline by the time its audio was captured.

const MAX_IN_FLIGHT = 5;
const MAX_FILE_SECONDS = 60;
const TIMELINE_SECONDS = 120;
const BUFFER_SECONDS = 10;
const CALIBRATION_SECONDS = 6;
const BASELINE_KEY = "spectromood-baseline";

const EMOJI = { angry: "😠", fearful: "😨", happy: "😄", sad: "😢", neutral: "😐" };
const emoji = (e) => EMOJI[e] || "🙂";

const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const state = {
  emotions: [],
  arousalLevels: [],
  results: [], // {t, emotion, probabilities, arousal}
  latestShown: 0,
  inFlight: 0,
  recording: false,
  ctx: null,
  stream: null,
  analyser: null,
  ring: null,
  ringPos: 0,
  ringFilled: 0, // samples currently held in the ring (capped at its length)
  totalSamples: 0, // samples received since the mic started (never capped)
  sampleRate: 48000,
  timer: null,
  baseline: null, // the speaker's calm reference features from /api/baseline
  baselineVariants: [],
  calibrating: false,
};

function log(msg, isError = false) {
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  if (isError) li.className = "err";
  $("log").prepend(li);
  while ($("log").children.length > 200) $("log").lastChild.remove();
}

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = `status ${kind || ""}`;
}

function emotionColor(e) {
  return css(`--${e}`) || css("--neutral");
}

// ---------- WAV encoding ----------
function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}

// ---------- Talking to the backend ----------
async function analyse(samples, sampleRate, capturedAt, label, useBaseline = false) {
  state.inFlight++;
  updatePending();
  const variant = $("variantSel").value;
  try {
    const resp = await fetch(`/api/analyze?variant=${encodeURIComponent(variant)}`, {
      method: "POST",
      headers: {
        "Content-Type": "audio/wav",
        // The baseline describes this microphone + speaker, so it is only sent for mic audio.
        ...(useBaseline && state.baseline ? { "X-Baseline": JSON.stringify(state.baseline) } : {}),
      },
      body: encodeWav(samples, sampleRate),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    if (data.status === "silence") {
      setVoice(false);
      log(`🤫 ${label}: silence (${data.level_dbfs} dBFS), not sent to Clef`);
      return;
    }
    if (data.status === "no_speech") {
      setVoice(false);
      log(`🔇 ${label}: no voice detected (only ${data.speech_seconds}s), not sent to Clef`);
      return;
    }
    setVoice(true);
    handleResult(data, capturedAt, label);
  } catch (err) {
    log(`${label}: ${err.message}`, true);
  } finally {
    state.inFlight--;
    updatePending();
  }
}

function setVoice(present) {
  $("voiceState").textContent = present ? "🗣️ voice" : "🔇 no voice: Clef idle";
  $("voiceState").classList.toggle("idle", !present);
}

function updatePending() {
  $("pending").textContent = `${state.inFlight} in flight`;
}

function handleResult(data, capturedAt, label) {
  state.results.push({ t: capturedAt, emotion: data.emotion, probabilities: data.probabilities, arousal: data.arousal });
  const lag = ((Date.now() - capturedAt) / 1000).toFixed(1);
  log(`${emoji(data.emotion)} ${label}: ${data.emotion} (Clef ${data.clef_latency_s}s, ${lag}s after capture)`);
  if (capturedAt < state.latestShown) return; // an older window finished late
  state.latestShown = capturedAt;

  $("currentEmoji").textContent = emoji(data.emotion);
  $("currentEmoji").classList.remove("pop");
  void $("currentEmoji").offsetWidth; // restart the pop animation
  $("currentEmoji").classList.add("pop");
  $("currentEmotion").textContent = data.emotion;
  $("currentEmotion").style.color = emotionColor(data.emotion);
  const conf = data.confidence != null ? ` · confidence ${(data.confidence * 100).toFixed(0)}%` : "";
  $("currentMeta").textContent =
    `${label} · audio captured ${lag}s ago · Clef ${data.clef_latency_s}s · ${data.model} / ${data.variant}${conf}`;
  renderBars(data.probabilities);
  renderArousal(data.arousal);
  if (data.image) {
    $("sentImage").src = data.image;
    $("sentImage").hidden = false;
    $("noImage").hidden = true;
  } else {
    $("sentImage").hidden = true;
    $("noImage").hidden = false;
    $("noImage").textContent = "This variant sends only text features, no image.";
  }
  const f = data.features;
  const fmt = (v, unit) => `${v > 0 ? "+" : ""}${v}${unit}`;
  $("prosody").textContent = data.baseline_used
    ? `vs your calm voice: pitch ${fmt(f.pitch_vs_baseline_st, " st")} · loudness ${fmt(f.level_vs_baseline_db, " dB")} · tilt ${fmt(f.tilt_vs_baseline_db, " dB")}`
    : `pitch ${data.prosody.pitch_median_hz} Hz · range ${f.pitch_range_st} st · ${f.onsets_per_s} onsets/s`;
}

function renderBars(probs) {
  const bars = $("bars");
  bars.replaceChildren();
  for (const e of state.emotions) {
    const p = probs[e] ?? 0;
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `<span></span><div class="bar-track"><div class="bar-fill"></div></div><span class="bar-val"></span>`;
    row.children[0].textContent = `${emoji(e)} ${e}`;
    row.querySelector(".bar-fill").style.background = emotionColor(e);
    row.querySelector(".bar-val").textContent = `${(p * 100).toFixed(0)}%`;
    bars.append(row);
    requestAnimationFrame(() => (row.querySelector(".bar-fill").style.width = `${p * 100}%`));
  }
}

function renderArousal(score) {
  const max = state.arousalLevels.length - 1;
  if (score == null || max <= 0) {
    $("arousalFill").style.width = "0";
    $("arousalLabel").textContent = "—";
    return;
  }
  $("arousalFill").style.width = `${(score / max) * 100}%`;
  $("arousalLabel").textContent = state.arousalLevels[Math.round(score)] ?? score.toFixed(1);
}

// ---------- Microphone ----------
async function startMic() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      // Keep raw dynamics: browser processing flattens the loudness cues we rely on.
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
  } catch (err) {
    log(`Microphone unavailable: ${err.message}`, true);
    return;
  }
  state.ctx = new AudioContext();
  state.sampleRate = state.ctx.sampleRate;
  await state.ctx.audioWorklet.addModule("/static/recorder-worklet.js");
  const source = state.ctx.createMediaStreamSource(state.stream);
  state.analyser = state.ctx.createAnalyser();
  state.analyser.fftSize = 1024;
  state.analyser.smoothingTimeConstant = 0;
  const recorder = new AudioWorkletNode(state.ctx, "recorder-processor");
  const mute = state.ctx.createGain();
  mute.gain.value = 0; // keep the graph pulled without playing the mic back
  source.connect(state.analyser);
  source.connect(recorder).connect(mute).connect(state.ctx.destination);

  state.ring = new Float32Array(state.sampleRate * BUFFER_SECONDS);
  state.ringPos = 0;
  state.ringFilled = 0;
  state.totalSamples = 0;
  recorder.port.onmessage = (ev) => pushSamples(ev.data);

  state.recording = true;
  $("micBtn").textContent = "⏹️ Stop microphone";
  $("micBtn").classList.add("recording");
  log(`Microphone started (${state.sampleRate} Hz)`);
  scheduleSends();
  drawLive();
}

function stopMic() {
  state.recording = false;
  clearInterval(state.timer);
  state.stream?.getTracks().forEach((t) => t.stop());
  state.ctx?.close();
  state.ctx = null;
  $("micBtn").textContent = "🎙️ Start microphone";
  $("micBtn").classList.remove("recording");
  $("meterFill").style.width = "0";
  log("Microphone stopped");
}

function pushSamples(chunk) {
  const ring = state.ring;
  for (let i = 0; i < chunk.length; i++) {
    ring[state.ringPos] = chunk[i];
    state.ringPos = (state.ringPos + 1) % ring.length;
  }
  state.ringFilled = Math.min(ring.length, state.ringFilled + chunk.length);
  state.totalSamples += chunk.length;
}

function lastSeconds(seconds) {
  const n = Math.min(Math.round(seconds * state.sampleRate), state.ringFilled);
  const out = new Float32Array(n);
  const start = (state.ringPos - n + state.ring.length) % state.ring.length;
  for (let i = 0; i < n; i++) out[i] = state.ring[(start + i) % state.ring.length];
  return out;
}

function scheduleSends() {
  clearInterval(state.timer);
  const hop = Number($("hopSel").value) * 1000;
  state.timer = setInterval(() => {
    if (!state.recording) return;
    const win = Number($("windowSel").value);
    if (state.ringFilled < win * state.sampleRate) return;
    if (state.inFlight >= MAX_IN_FLIGHT) {
      log("Skipped a window: Clef is still busy with earlier ones");
      return;
    }
    if (state.calibrating) return;
    analyse(lastSeconds(win), state.sampleRate, Date.now(), "mic", true);
  }, hop);
}

// ---------- Live spectrogram (local, instant) ----------
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function magma(v) {
  // Compact approximation of the magma colormap (v in 0..1).
  const stops = [[0, 0, 4], [80, 18, 123], [182, 54, 121], [251, 136, 97], [252, 253, 191]];
  const x = Math.max(0, Math.min(1, v)) * (stops.length - 1);
  const i = Math.min(Math.floor(x), stops.length - 2);
  const f = x - i;
  return stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f));
}

function drawLive() {
  if (!state.recording) return;
  const canvas = $("liveSpec");
  fitCanvas(canvas);
  const g = canvas.getContext("2d");
  const bins = new Uint8Array(state.analyser.frequencyBinCount);
  state.analyser.getByteFrequencyData(bins);

  // Scroll left by 2px and paint the newest column on the right (0-8 kHz shown).
  const colW = 2;
  g.drawImage(canvas, -colW, 0);
  const maxBin = Math.floor((8000 / (state.sampleRate / 2)) * bins.length);
  const col = g.createImageData(colW, canvas.height);
  for (let y = 0; y < canvas.height; y++) {
    const bin = Math.floor(((canvas.height - 1 - y) / canvas.height) * maxBin);
    const [r, gg, b] = magma(bins[bin] / 255);
    for (let x = 0; x < colW; x++) {
      const o = (y * colW + x) * 4;
      col.data[o] = r;
      col.data[o + 1] = gg;
      col.data[o + 2] = b;
      col.data[o + 3] = 255;
    }
  }
  g.putImageData(col, canvas.width - colW, 0);

  const td = new Float32Array(state.analyser.fftSize);
  state.analyser.getFloatTimeDomainData(td);
  const rms = Math.sqrt(td.reduce((s, x) => s + x * x, 0) / td.length);
  const db = 20 * Math.log10(Math.max(rms, 1e-6));
  $("level").textContent = `${db.toFixed(0)} dBFS`;
  $("meterFill").style.width = `${Math.max(0, Math.min(100, ((db + 60) / 60) * 100))}%`;

  requestAnimationFrame(drawLive);
}

// ---------- Timeline ----------
function drawTimeline() {
  const canvas = $("timeline");
  fitCanvas(canvas);
  const g = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.width, H = canvas.height;
  const pad = { l: 36 * dpr, r: 8 * dpr, t: 30 * dpr, b: 22 * dpr };
  g.clearRect(0, 0, W, H);

  const now = Date.now();
  const x = (t) => pad.l + ((t - (now - TIMELINE_SECONDS * 1000)) / (TIMELINE_SECONDS * 1000)) * (W - pad.l - pad.r);
  const y = (p) => pad.t + (1 - p) * (H - pad.t - pad.b);

  g.font = `${11 * dpr}px system-ui, sans-serif`;
  g.fillStyle = css("--muted");
  g.strokeStyle = css("--border");
  g.lineWidth = 1;
  for (const p of [0, 0.5, 1]) {
    g.beginPath();
    g.moveTo(pad.l, y(p));
    g.lineTo(W - pad.r, y(p));
    g.stroke();
    g.fillText(`${p * 100}%`, 4 * dpr, y(p) + 4 * dpr);
  }
  for (let s = 0; s <= TIMELINE_SECONDS; s += 30) {
    g.fillText(s === 0 ? "now" : `-${s}s`, x(now - s * 1000) - 12 * dpr, H - 6 * dpr);
  }

  const pts = state.results.filter((r) => r.t > now - TIMELINE_SECONDS * 1000).sort((a, b) => a.t - b.t);
  g.lineWidth = 2 * dpr;
  for (const e of state.emotions) {
    g.strokeStyle = g.fillStyle = emotionColor(e);
    g.beginPath();
    pts.forEach((r, i) => {
      const px = x(r.t), py = y(r.probabilities[e] ?? 0);
      i ? g.lineTo(px, py) : g.moveTo(px, py);
    });
    g.stroke();
    for (const r of pts) {
      g.beginPath();
      g.arc(x(r.t), y(r.probabilities[e] ?? 0), 2.5 * dpr, 0, Math.PI * 2);
      g.fill();
    }
  }

  // Emoji strip: the winning emotion of each result, skipping ones that would overlap.
  g.font = `${16 * dpr}px system-ui, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif`;
  g.textAlign = "center";
  let lastX = -Infinity;
  for (const r of pts) {
    const px = x(r.t);
    if (px - lastX < 20 * dpr) continue;
    g.fillText(emoji(r.emotion), px, 20 * dpr);
    lastX = px;
  }
  g.textAlign = "start";
}

// ---------- Speaker baseline ----------
function setBaseline(baseline) {
  state.baseline = baseline;
  try {
    if (baseline) localStorage.setItem(BASELINE_KEY, JSON.stringify(baseline));
    else localStorage.removeItem(BASELINE_KEY);
  } catch {
    /* storage unavailable: the baseline just lasts for this page */
  }
  $("baselineStatus").textContent = baseline
    ? `Calibrated · your calm pitch ${Math.round(baseline.pitch_median_hz)} Hz`
    : "Not calibrated";
  $("clearBaseline").hidden = !baseline;
  updateVariantHint();
}

function updateVariantHint() {
  const needs = state.baselineVariants.includes($("variantSel").value);
  $("variantHint").textContent =
    needs && !state.baseline ? "This input compares with your calm voice: calibrate first for best results." : "";
}

async function calibrate() {
  if (!state.recording) await startMic();
  if (!state.recording) return;
  state.calibrating = true;
  $("calibrateBtn").disabled = true;
  // Count with the uncapped total: the ring's fill level stops growing once it is full.
  const startTotal = state.totalSamples;
  for (let left = CALIBRATION_SECONDS; left > 0; left--) {
    $("baselineStatus").textContent = `Speak calmly and normally… ${left}s`;
    await new Promise((r) => setTimeout(r, 1000));
  }
  const recorded = Math.min(state.totalSamples - startTotal, CALIBRATION_SECONDS * state.sampleRate);
  try {
    const resp = await fetch("/api/baseline", {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: encodeWav(lastSeconds(recorded / state.sampleRate), state.sampleRate),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
    setBaseline(data.baseline);
    log(`🧘 Calibrated on ${data.speech_seconds ?? data.seconds}s of calm speech`);
  } catch (err) {
    log(`Calibration failed: ${err.message}`, true);
    setBaseline(state.baseline);
    $("baselineStatus").textContent = `❌ ${err.message}. Try again`;
  } finally {
    state.calibrating = false;
    $("calibrateBtn").disabled = false;
  }
}

// ---------- File analysis ----------
async function analyseFile(file) {
  const ctx = new AudioContext();
  try {
    const audio = await ctx.decodeAudioData(await file.arrayBuffer());
    const mono = new Float32Array(audio.length);
    for (let c = 0; c < audio.numberOfChannels; c++) {
      const ch = audio.getChannelData(c);
      for (let i = 0; i < ch.length; i++) mono[i] += ch[i] / audio.numberOfChannels;
    }
    const sr = audio.sampleRate;
    const total = Math.min(audio.duration, MAX_FILE_SECONDS);
    const win = Number($("windowSel").value);
    // Short files go as one request; longer ones in consecutive windows.
    const segments = [];
    if (total <= 15) segments.push([0, total]);
    else for (let s = 0; s + 1 <= total; s += win) segments.push([s, Math.min(s + win, total)]);
    log(`${file.name}: ${audio.duration.toFixed(1)}s, ${segments.length} request(s)`);
    for (const [a, b] of segments) {
      while (state.inFlight >= MAX_IN_FLIGHT) await new Promise((r) => setTimeout(r, 300));
      analyse(mono.slice(Math.floor(a * sr), Math.floor(b * sr)), sr, Date.now(), `${file.name} @${a.toFixed(0)}s`);
    }
  } catch (err) {
    log(`${file.name}: could not decode (${err.message})`, true);
  } finally {
    ctx.close();
  }
}

// ---------- Boot ----------
async function boot() {
  let health;
  try {
    health = await (await fetch("/api/health")).json();
  } catch (err) {
    setStatus("Backend unreachable", "err");
    return;
  }
  state.emotions = health.emotions;
  state.arousalLevels = health.arousal_levels;
  state.baselineVariants = health.baseline_variants || [];
  for (const v of health.variants) {
    const opt = new Option(v, v, false, v === health.default_variant);
    $("variantSel").append(opt);
  }
  $("legend").replaceChildren(
    ...state.emotions.map((e) => {
      const s = document.createElement("span");
      s.textContent = `${emoji(e)} ${e}`;
      s.style.setProperty("--c", emotionColor(e));
      return s;
    }),
  );
  renderBars({});

  if (!health.ok) {
    setStatus("Clef not configured", "err");
    log(health.error || "Clef not configured", true);
    return;
  }
  setStatus(`Connected · ${health.model}`, "ok");
  $("micBtn").disabled = false;
  $("fileInput").disabled = false;

  $("micBtn").addEventListener("click", () => (state.recording ? stopMic() : startMic()));
  $("calibrateBtn").disabled = false;
  $("calibrateBtn").addEventListener("click", calibrate);
  $("clearBaseline").addEventListener("click", () => setBaseline(null));
  $("variantSel").addEventListener("change", updateVariantHint);
  let saved = null;
  try {
    saved = JSON.parse(localStorage.getItem(BASELINE_KEY) || "null");
  } catch {
    saved = null;
  }
  setBaseline(saved);
  $("hopSel").addEventListener("change", () => state.recording && scheduleSends());
  $("fileInput").addEventListener("change", (ev) => {
    const f = ev.target.files[0];
    if (f) analyseFile(f);
    ev.target.value = "";
  });
  setInterval(drawTimeline, 500);
}

boot();
