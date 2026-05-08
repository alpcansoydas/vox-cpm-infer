import argparse
import inspect
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from funasr import AutoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


UPLOAD_DIR = Path(tempfile.gettempdir()) / "voxcpm_fastapi_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class VoxCPMFastAPIDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        logger.info("Running on device: %s", self.device)

        self.asr_model_id = "iic/SenseVoiceSmall"
        logger.info("Loading ASR model: %s", self.asr_model_id)
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device="cuda:0" if self.device == "cuda" else "cpu",
        )
        logger.info("ASR model loaded successfully.")

        logger.info("Loading VoxCPM model before starting FastAPI: %s", self.model_id)
        self.voxcpm_model: voxcpm.VoxCPM = voxcpm.VoxCPM.from_pretrained(
            self.model_id,
            optimize=False,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        logger.info("VoxCPM model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        if not self.asr_model:
            return ""
        res = self.asr_model.generate(input=audio_path, language="auto", use_itn=True)
        return res[0]["text"].split("|>")[-1]

    def _supported_generate_kwargs(self, generate_kwargs: dict) -> dict:
        generate_fn = getattr(self.voxcpm_model, "_generate", None)
        if generate_fn is None:
            return generate_kwargs
        try:
            signature = inspect.signature(generate_fn)
        except (TypeError, ValueError):
            return generate_kwargs
        if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return generate_kwargs
        supported = set(signature.parameters)
        filtered = {key: value for key, value in generate_kwargs.items() if key in supported}
        dropped = sorted(set(generate_kwargs) - set(filtered))
        if dropped:
            logger.warning(
                "Ignoring unsupported VoxCPM generation option(s) for this installed backend: %s",
                ", ".join(dropped),
            )
        return filtered

    def build_generate_kwargs(self, payload: dict) -> dict:
        text = (payload.get("text") or "").strip()
        if not text:
            raise ValueError("Please input text to synthesize.")

        use_prompt_text = bool(payload.get("use_prompt_text"))
        control = "" if use_prompt_text else (payload.get("control_instruction") or "").strip()
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        reference_audio_path = payload.get("reference_wav_path") or None
        prompt_audio_path = payload.get("prompt_wav_path") or None
        prompt_text = (payload.get("prompt_text") or "").strip() or None
        if prompt_text and prompt_audio_path is None and reference_audio_path:
            prompt_audio_path = reference_audio_path

        generate_kwargs = {
            "text": final_text,
            "reference_wav_path": reference_audio_path,
            "cfg_value": float(payload.get("cfg_value", 2.0)),
            "inference_timesteps": int(payload.get("inference_timesteps", 10)),
            "min_len": int(payload.get("min_len", 2)),
            "max_len": int(payload.get("max_len", 4096)),
            "normalize": bool(payload.get("normalize", False)),
            "denoise": bool(payload.get("denoise", False)),
            "retry_badcase": bool(payload.get("retry_badcase", True)),
            "retry_badcase_max_times": int(payload.get("retry_badcase_max_times", 3)),
            "retry_badcase_ratio_threshold": float(payload.get("retry_badcase_ratio_threshold", 6.0)),
            "trim_silence_vad": bool(payload.get("trim_silence_vad", False)),
            "streaming_prefix_len": int(payload.get("streaming_prefix_len", 4)),
        }
        if prompt_text and prompt_audio_path:
            generate_kwargs["prompt_wav_path"] = prompt_audio_path
            generate_kwargs["prompt_text"] = prompt_text
        return self._supported_generate_kwargs(generate_kwargs)

    def stream_pcm(self, payload: dict):
        generate_kwargs = self.build_generate_kwargs(payload)
        for chunk in self.voxcpm_model.generate_streaming(**generate_kwargs):
            wav = np.asarray(chunk, dtype=np.float32)
            if wav.size == 0:
                continue
            if wav.ndim > 1:
                wav = wav.reshape(-1)
            yield np.ascontiguousarray(wav)


def create_app(demo: VoxCPMFastAPIDemo) -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(HTML)

    @app.post("/upload")
    async def upload_audio(file: UploadFile = File(...)):
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
        with path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return JSONResponse({"path": str(path)})

    @app.post("/asr")
    async def asr(payload: dict):
        audio_path = payload.get("path")
        if not audio_path:
            return JSONResponse({"text": ""})
        try:
            return JSONResponse({"text": demo.transcribe(audio_path)})
        except Exception as exc:
            logger.warning("ASR failed: %s", exc)
            return JSONResponse({"text": ""}, status_code=500)

    @app.post("/stream")
    async def stream(payload: dict):
        sample_rate = int(demo.voxcpm_model.tts_model.sample_rate)

        def iter_pcm():
            start = time.perf_counter()
            for wav in demo.stream_pcm(payload):
                yield wav.tobytes()
            logger.info("Finished HTTP PCM stream in %.3fs", time.perf_counter() - start)

        return StreamingResponse(
            iter_pcm(),
            media_type="application/octet-stream",
            headers={"X-Sample-Rate": str(sample_rate)},
        )

    return app


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VoxCPM Fast Stream</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --panel: #ffffff;
      --text: #18202f;
      --muted: #687386;
      --line: #dbe1ea;
      --accent: #1769e0;
      --accent-strong: #0f54b7;
      --danger: #bd2a2a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 740;
      letter-spacing: 0;
    }
    .status {
      min-width: 180px;
      text-align: right;
      color: var(--muted);
      font-size: 14px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
      gap: 18px;
      align-items: start;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 650;
      margin: 0 0 7px;
      color: #273246;
    }
    textarea, input[type="text"], input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      font: inherit;
      background: #fff;
      color: var(--text);
      outline: none;
    }
    textarea:focus, input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(23, 105, 224, 0.12);
    }
    textarea {
      resize: vertical;
      min-height: 82px;
    }
    .field { margin-bottom: 14px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .row {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .check {
      display: flex;
      gap: 8px;
      align-items: center;
      color: #273246;
      font-size: 14px;
      margin: 8px 0;
    }
    .check input { width: 16px; height: 16px; }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font-weight: 720;
      font-size: 15px;
      padding: 11px 16px;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button.secondary {
      background: #eef3fb;
      color: #1d3354;
    }
    button.secondary:hover { background: #e2ebf8; }
    button.danger {
      background: var(--danger);
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
    }
    input[type="file"] {
      width: 100%;
      border: 1px dashed #b8c2d1;
      padding: 10px;
      border-radius: 6px;
      background: #fbfcfe;
    }
    input[type="range"] { width: 100%; }
    .meter {
      height: 96px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: linear-gradient(180deg, #fbfcff, #f0f4fa);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-size: 14px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfe;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong {
      font-size: 16px;
    }
    .log {
      height: 170px;
      overflow: auto;
      background: #101827;
      color: #d8e0ee;
      border-radius: 6px;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      margin-top: 12px;
    }
    @media (max-width: 850px) {
      main { width: min(100vw - 20px, 720px); margin: 16px auto; }
      .layout, .grid, .metrics { grid-template-columns: 1fr; }
      header { align-items: flex-start; flex-direction: column; }
      .status { text-align: left; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>VoxCPM Fast Stream</h1>
      <div class="status" id="status">Ready</div>
    </header>

    <div class="layout">
      <section>
        <div class="field">
          <label for="text">Target text</label>
          <textarea id="text">VoxCPM2 is streaming through a low-latency Web Audio pipeline.</textarea>
        </div>
        <div class="field">
          <label for="control">Control instruction</label>
          <textarea id="control" placeholder="Young woman, gentle and warm voice, natural pace"></textarea>
        </div>
        <div class="grid">
          <div class="field">
            <label for="reference">Reference audio</label>
            <input id="reference" type="file" accept="audio/*" />
          </div>
          <div class="field">
            <label for="promptAudio">Prompt audio</label>
            <input id="promptAudio" type="file" accept="audio/*" />
          </div>
        </div>
        <label class="check">
          <input id="usePromptText" type="checkbox" />
          Use prompt transcript for ultimate cloning
        </label>
        <div class="field">
          <label for="promptText">Prompt transcript</label>
          <textarea id="promptText" placeholder="Transcript of the prompt or reference audio"></textarea>
        </div>
        <div class="row">
          <button id="asrBtn" class="secondary" type="button">Transcribe Reference</button>
        </div>
      </section>

      <section>
        <div class="meter" id="meter">Audio will start as soon as the first PCM chunk arrives.</div>
        <div class="metrics">
          <div class="metric"><span>First audio</span><strong id="firstLatency">-</strong></div>
          <div class="metric"><span>Total</span><strong id="totalTime">-</strong></div>
          <div class="metric"><span>Queued audio</span><strong id="queuedAudio">0 ms</strong></div>
        </div>
        <div class="field" style="margin-top:14px">
          <label for="playbackBuffer">Playback buffer: <span id="bufferValue">80</span> ms</label>
          <input id="playbackBuffer" type="range" min="20" max="500" value="80" step="10" />
        </div>
        <div class="row">
          <button id="generateBtn" type="button">Generate</button>
          <button id="stopBtn" class="danger" type="button" disabled>Stop</button>
          <button id="resumeBtn" class="secondary" type="button">Resume Audio</button>
        </div>
        <div class="log" id="log"></div>
      </section>
    </div>

    <section style="margin-top:18px">
      <div class="grid">
        <div class="field">
          <label for="cfg">CFG value</label>
          <input id="cfg" type="number" min="1" max="3" value="2.0" step="0.1" />
        </div>
        <div class="field">
          <label for="steps">Inference steps</label>
          <input id="steps" type="number" min="1" max="50" value="10" step="1" />
        </div>
        <div class="field">
          <label for="minLen">Min length</label>
          <input id="minLen" type="number" min="0" max="100" value="2" step="1" />
        </div>
        <div class="field">
          <label for="maxLen">Max length</label>
          <input id="maxLen" type="number" min="16" max="8192" value="4096" step="16" />
        </div>
        <div class="field">
          <label for="prefixLen">Streaming prefix length</label>
          <input id="prefixLen" type="number" min="1" max="16" value="4" step="1" />
        </div>
        <div class="field">
          <label for="retryRatio">Retry ratio threshold</label>
          <input id="retryRatio" type="number" min="1" max="20" value="6.0" step="0.5" />
        </div>
      </div>
      <div class="row">
        <label class="check"><input id="normalize" type="checkbox" /> Normalize text</label>
        <label class="check"><input id="denoise" type="checkbox" /> Denoise prompt/reference</label>
        <label class="check"><input id="retry" type="checkbox" checked /> Retry badcase</label>
        <label class="check"><input id="trimSilence" type="checkbox" /> Trim silence VAD</label>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const logEl = $("log");
    const state = {
      abortController: null,
      audioCtx: null,
      nextStartTime: 0,
      sampleRate: 24000,
      startedAt: 0,
      referencePath: "",
      promptPath: "",
      chunkCount: 0,
      sourceNodes: [],
      pendingPcmBytes: new Uint8Array(0)
    };

    function log(message) {
      const line = `[${new Date().toLocaleTimeString()}] ${message}`;
      logEl.textContent += `${line}\n`;
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setBusy(isBusy) {
      $("generateBtn").disabled = isBusy;
      $("stopBtn").disabled = !isBusy;
      $("status").textContent = isBusy ? "Generating" : "Ready";
    }

    function playbackLeadSeconds() {
      return Number($("playbackBuffer").value) / 1000;
    }

    $("playbackBuffer").addEventListener("input", () => {
      $("bufferValue").textContent = $("playbackBuffer").value;
    });

    async function ensureAudioContext() {
      if (!state.audioCtx) {
        state.audioCtx = new AudioContext({ latencyHint: "interactive" });
      }
      if (state.audioCtx.state !== "running") {
        await state.audioCtx.resume();
      }
      return state.audioCtx;
    }

    function resetPlaybackClock() {
      const ctx = state.audioCtx;
      state.nextStartTime = ctx ? ctx.currentTime + playbackLeadSeconds() : 0;
      state.chunkCount = 0;
      state.sourceNodes.forEach((node) => {
        try { node.stop(); } catch (_) {}
      });
      state.sourceNodes = [];
      state.pendingPcmBytes = new Uint8Array(0);
    }

    function schedulePcmBytes(bytes) {
      const ctx = state.audioCtx;
      if (!ctx) return;
      const pcm = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / Float32Array.BYTES_PER_ELEMENT);
      if (!pcm.length) return;

      const audioBuffer = ctx.createBuffer(1, pcm.length, state.sampleRate);
      audioBuffer.copyToChannel(pcm, 0);

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);

      const minStart = ctx.currentTime + playbackLeadSeconds();
      const startAt = Math.max(state.nextStartTime, minStart);
      source.start(startAt);
      state.nextStartTime = startAt + audioBuffer.duration;
      state.chunkCount += 1;
      state.sourceNodes.push(source);
      source.onended = () => {
        state.sourceNodes = state.sourceNodes.filter((node) => node !== source);
      };

      const queuedMs = Math.max(0, state.nextStartTime - ctx.currentTime) * 1000;
      $("queuedAudio").textContent = `${Math.round(queuedMs)} ms`;
      $("meter").textContent = `Playing chunk ${state.chunkCount}`;
    }

    function processPcmChunk(chunk) {
      let bytes = chunk;
      if (state.pendingPcmBytes.byteLength) {
        bytes = new Uint8Array(state.pendingPcmBytes.byteLength + chunk.byteLength);
        bytes.set(state.pendingPcmBytes, 0);
        bytes.set(chunk, state.pendingPcmBytes.byteLength);
        state.pendingPcmBytes = new Uint8Array(0);
      }

      const completeLength = bytes.byteLength - (bytes.byteLength % Float32Array.BYTES_PER_ELEMENT);
      if (completeLength > 0) {
        schedulePcmBytes(bytes.subarray(0, completeLength));
      }
      if (completeLength < bytes.byteLength) {
        state.pendingPcmBytes = bytes.slice(completeLength);
      }
    }

    async function uploadFile(input) {
      if (!input.files || !input.files[0]) return "";
      const form = new FormData();
      form.append("file", input.files[0]);
      const res = await fetch("/upload", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const data = await res.json();
      return data.path;
    }

    async function syncUploads() {
      state.referencePath = await uploadFile($("reference"));
      state.promptPath = await uploadFile($("promptAudio"));
    }

    $("asrBtn").addEventListener("click", async () => {
      try {
        $("asrBtn").disabled = true;
        if (!state.referencePath) {
          state.referencePath = await uploadFile($("reference"));
        }
        if (!state.referencePath) {
          log("Choose a reference audio file first.");
          return;
        }
        log("Running ASR...");
        const res = await fetch("/asr", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: state.referencePath })
        });
        const data = await res.json();
        $("promptText").value = data.text || "";
        $("usePromptText").checked = true;
        log("ASR finished.");
      } catch (err) {
        log(err.message);
      } finally {
        $("asrBtn").disabled = false;
      }
    });

    $("resumeBtn").addEventListener("click", async () => {
      await ensureAudioContext();
      log("Audio context resumed.");
    });

    $("stopBtn").addEventListener("click", () => {
      if (state.abortController) state.abortController.abort();
      resetPlaybackClock();
      setBusy(false);
      log("Stopped.");
    });

    $("generateBtn").addEventListener("click", async () => {
      try {
        setBusy(true);
        $("firstLatency").textContent = "-";
        $("totalTime").textContent = "-";
        $("queuedAudio").textContent = "0 ms";
        $("meter").textContent = "Connecting...";
        await ensureAudioContext();
        resetPlaybackClock();
        await syncUploads();

        state.startedAt = performance.now();

        const payload = {
          text: $("text").value,
          control_instruction: $("control").value,
          reference_wav_path: state.referencePath,
          prompt_wav_path: state.promptPath,
          use_prompt_text: $("usePromptText").checked,
          prompt_text: $("promptText").value,
          cfg_value: Number($("cfg").value),
          inference_timesteps: Number($("steps").value),
          min_len: Number($("minLen").value),
          max_len: Number($("maxLen").value),
          normalize: $("normalize").checked,
          denoise: $("denoise").checked,
          retry_badcase: $("retry").checked,
          retry_badcase_max_times: 3,
          retry_badcase_ratio_threshold: Number($("retryRatio").value),
          trim_silence_vad: $("trimSilence").checked,
          streaming_prefix_len: Number($("prefixLen").value)
        };

        state.abortController = new AbortController();
        log("Streaming request sent.");
        const response = await fetch("/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: state.abortController.signal
        });
        if (!response.ok || !response.body) {
          throw new Error(`Stream failed: HTTP ${response.status}`);
        }

        state.sampleRate = Number(response.headers.get("X-Sample-Rate")) || state.sampleRate;
        $("meter").textContent = `Streaming at ${state.sampleRate} Hz`;
        log(`HTTP PCM stream started at ${state.sampleRate} Hz.`);

        const reader = response.body.getReader();
        let gotFirstChunk = false;
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          if (!value || value.byteLength === 0) continue;
          if (!gotFirstChunk) {
            gotFirstChunk = true;
            $("firstLatency").textContent = `${((performance.now() - state.startedAt) / 1000).toFixed(3)} s`;
          }
          processPcmChunk(value);
        }

        if (state.pendingPcmBytes.byteLength) {
          log(`Dropped ${state.pendingPcmBytes.byteLength} trailing PCM byte(s).`);
          state.pendingPcmBytes = new Uint8Array(0);
        }
        $("totalTime").textContent = `${((performance.now() - state.startedAt) / 1000).toFixed(3)} s`;
        log("Generation complete.");
        state.abortController = null;
        setBusy(false);
      } catch (err) {
        if (err.name === "AbortError") {
          log("Stream aborted.");
        } else {
          log(err.message);
        }
        state.abortController = null;
        setBusy(false);
      }
    });
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="openbmb/VoxCPM2")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8808)
    args = parser.parse_args()

    demo = VoxCPMFastAPIDemo(model_id=args.model_id)
    app = create_app(demo)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
