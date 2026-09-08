"""
Unified Server for Voice-Activated Edge Device (ESP32-S3 + KWS + Whisper).
Supports both:
  1. USB Serial (/dev/ttyACM* or COM7 @ 921600 baud) for prototype demo.
  2. Wi-Fi WebSockets (ws://host:8080/stream) for final deployment.
"""

import asyncio
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8080
SERIAL_PORT = os.environ.get("ESP32_SERIAL_PORT", "COM7")
SERIAL_BAUD = 921600  # High-speed baud rate for 16kHz PCM streaming

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit PCM (2 bytes/sample)
CHANNELS = 1

VAD_CHUNK_SAMPLES = 512
VAD_CHUNK_BYTES = VAD_CHUNK_SAMPLES * SAMPLE_WIDTH

# Responsive VAD settings
VAD_THRESHOLD = 0.35          # Increased sensitivity for edge microphones
VAD_MIN_SILENCE_MS = 1200     # 1.2s continuous silence => finalize utterance
VAD_SPEECH_PAD_MS = 60        # Pad speech boundaries to preserve start/end syllables

MAX_UTTERANCE_SEC = 7.0       # Hard cap on continuous speech
NO_SPEECH_TIMEOUT_SEC = 4.0   # If no speech detected within 4s of wake word, finalize

WHISPER_MODEL_SIZE = "base.en"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_CPU_THREADS = os.cpu_count() or 4

STATS_FILE = os.path.join(os.path.dirname(__file__), "esp32_stats.jsonl")
EXECUTOR = ThreadPoolExecutor(max_workers=3)

# --------------------------------------------------------------------------
# Global Shared State & Telemetry
# --------------------------------------------------------------------------
latest_telemetry: Dict[str, Any] = {
    "cpu_percent": 0,
    "cpu0_percent": 0,
    "cpu1_percent": 0,
    "free_heap": 262144,
    "min_free_heap": 262144,
    "used_heap": 0,
    "uptime_ms": 0,
    "mic_peak": 0,
    "raw_hex": "0x00000000",
    "last_updated": 0,
}

latest_metrics: Dict[str, Any] = {
    "state": "IDLE",
    "transcript": "",
    "audio_duration_sec": 0.0,
    "whisper_latency_ms": 0.0,
    "vad_silence_latency_ms": 0.0,
    "total_e2e_latency_ms": 0.0,
    "timestamp": 0,
}

transcript_history: List[Dict[str, Any]] = []
event_subscribers: List[asyncio.Queue] = []
is_running = True

# --------------------------------------------------------------------------
# Model Loading
# --------------------------------------------------------------------------
print(f"[startup] Loading Faster-Whisper ({WHISPER_MODEL_SIZE}, {WHISPER_COMPUTE_TYPE}) ...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE,
    device="cpu",
    compute_type=WHISPER_COMPUTE_TYPE,
    cpu_threads=WHISPER_CPU_THREADS,
)
print("[startup] Faster-Whisper loaded.")

print("[startup] Loading Silero VAD ...")
silero_model = load_silero_vad(onnx=True)
print("[startup] Silero VAD loaded.")

app = FastAPI(title="ESP32-S3 Voice Assistant Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Session State & Helpers
# --------------------------------------------------------------------------
class Session:
    def __init__(self):
        self.state = "IDLE"
        self.audio_buffer = bytearray()
        self.vad_leftover = bytearray()
        self.vad_iterator = VADIterator(
            silero_model,
            threshold=VAD_THRESHOLD,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )
        self.wake_detected_time: float = 0.0
        self.speech_started: bool = False
        self.last_speech_time: float = 0.0
        self.chunks_received: int = 0
        self.max_peak: float = 0.0

    def reset(self):
        self.state = "IDLE"
        self.audio_buffer = bytearray()
        self.vad_leftover = bytearray()
        self.vad_iterator.reset_states()
        self.wake_detected_time = 0.0
        self.speech_started = False
        self.last_speech_time = 0.0
        self.chunks_received = 0
        self.max_peak = 0.0


def pcm_bytes_to_float32(chunk_bytes: bytes) -> np.ndarray:
    n = len(chunk_bytes) // 2
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    samples = struct.unpack(f"<{n}h", chunk_bytes[:n * 2])
    return np.array(samples, dtype=np.float32) / 32768.0


def transcribe_sync(audio_bytes: bytes) -> str:
    audio_np = pcm_bytes_to_float32(audio_bytes)
    if len(audio_np) == 0:
        return ""

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    max_amp = float(np.max(np.abs(audio_np)))

    # Only reject if audio is virtually pure zeros / dead mic (< 45 LSBs out of 32768)
    if max_amp < 0.0015 and rms < 0.0003:
        return "(Silence - No audible speech / check mic connection)"

    # Safe Peak Normalization: scale up quiet audio up to 0.85 peak without ANY clipping distortion
    if 0.005 < max_amp < 0.85:
        gain = min(3.0, 0.85 / max_amp)
        audio_np = audio_np * gain

    # Whisper transcription with repetition penalty and compression filter to eliminate repetition loops
    segments, _ = whisper_model.transcribe(
        audio_np,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.2,
        repetition_penalty=1.15,
    )
    res = " ".join(seg.text.strip() for seg in segments).strip()

    # Fallback without vad_filter if internal VAD was overly aggressive on soft voices
    if not res:
        segments, _ = whisper_model.transcribe(
            audio_np,
            language="en",
            beam_size=3,
            vad_filter=False,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.2,
        )
        res = " ".join(seg.text.strip() for seg in segments).strip()

    return res


def append_stats_line(record: dict):
    with open(STATS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


async def broadcast_event(data: dict):
    payload = f"data: {json.dumps(data)}\n\n"
    for q in list(event_subscribers):
        try:
            await q.put(payload)
        except Exception:
            event_subscribers.remove(q)


# --------------------------------------------------------------------------
# Pipeline Processor
# --------------------------------------------------------------------------
async def process_start_event(session: Session, source="WS"):
    session.reset()
    session.state = "STREAMING"
    session.wake_detected_time = time.perf_counter()
    latest_metrics["state"] = "STREAMING"
    print(f"\n[state] {source}: IDLE -> STREAMING (Listening...)")
    await broadcast_event({"type": "state", "state": "STREAMING", "source": source})


async def finalize_session(session: Session, stop_callback, loop, reason="silence"):
    if session.state != "STREAMING":
        return

    print(f"\n[vad] Finalizing utterance (reason: {reason}, peak: {session.max_peak:.3f})...")
    latest_metrics["state"] = "TRANSCRIBING"
    await broadcast_event({"type": "state", "state": "TRANSCRIBING"})

    # Send stop signal to ESP32
    await stop_callback()

    audio_bytes = bytes(session.audio_buffer)
    duration_sec = len(audio_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH)

    # Save audio to last_utterance.wav
    try:
        import wave
        wav_file = os.path.join(os.path.dirname(__file__), "last_utterance.wav")
        with wave.open(wav_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
    except Exception as ex:
        print(f"[warn] Could not save WAV file: {ex}")

    if duration_sec < 0.4:
        print("[transcript] (Audio too brief, skipping Whisper)")
        session.reset()
        latest_metrics["state"] = "IDLE"
        await broadcast_event({"type": "state", "state": "IDLE"})
        return

    t_whisper_start = time.perf_counter()
    transcript = await loop.run_in_executor(EXECUTOR, transcribe_sync, audio_bytes)
    t_whisper_end = time.perf_counter()

    whisper_ms = (t_whisper_end - t_whisper_start) * 1000.0
    total_e2e_ms = (t_whisper_end - session.wake_detected_time) * 1000.0

    # Keyword Spotting Verification & Command Extraction for 'Ankit'
    import re
    kws_detected = bool(re.search(r"\b(hey\s+)?(ankit|ankith|an\s+kit|un\s*kit)\b", transcript, re.IGNORECASE))
    if kws_detected:
        clean_command = re.sub(r"^(hey\s+)?(ankit|ankith|an\s+kit|un\s*kit)[\s,.:;!?-]*", "", transcript, flags=re.IGNORECASE).strip()
        display_text = f"🎯 [Wake Word: 'Ankit' Verified] ➔ \"{clean_command or transcript}\""
    else:
        display_text = transcript or "(No audible speech recognized)"

    result_entry = {
        "timestamp": time.time(),
        "transcript": display_text,
        "raw_transcript": transcript,
        "kws_verified": kws_detected,
        "audio_duration_sec": round(duration_sec, 2),
        "whisper_latency_ms": round(whisper_ms, 1),
        "total_e2e_latency_ms": round(total_e2e_ms, 1),
        "esp32_cpu": latest_telemetry.get("cpu_percent", 0),
        "esp32_free_heap_kb": round(latest_telemetry.get("free_heap", 0) / 1024, 1),
    }

    latest_metrics.update(result_entry)
    latest_metrics["state"] = "IDLE"
    transcript_history.append(result_entry)

    print(f"[transcript] {transcript!r}")
    print(f"[metrics] Audio: {duration_sec:.2f}s | Whisper: {whisper_ms:.1f}ms | Total: {total_e2e_ms:.1f}ms")
    print("[state] STREAMING -> IDLE (Ready for next trigger)\n")

    loop.run_in_executor(EXECUTOR, append_stats_line, result_entry)
    await broadcast_event({"type": "result", "data": result_entry})
    await broadcast_event({"type": "state", "state": "IDLE"})

    session.reset()


async def process_audio_chunk(session: Session, chunk: bytes, stop_callback, loop):
    if session.state != "STREAMING":
        return

    session.chunks_received += 1
    session.audio_buffer.extend(chunk)
    session.vad_leftover.extend(chunk)

    if session.chunks_received % 10 == 0:
        print(".", end="", flush=True)

    while len(session.vad_leftover) >= VAD_CHUNK_BYTES:
        vad_chunk = bytes(session.vad_leftover[:VAD_CHUNK_BYTES])
        del session.vad_leftover[:VAD_CHUNK_BYTES]

        vad_input = pcm_bytes_to_float32(vad_chunk)
        if len(vad_input) > 0:
            peak = float(np.max(np.abs(vad_input)))
            if peak > session.max_peak:
                session.max_peak = peak

            # Acoustic Energy Voice Activity Detection (peak >= 0.04 = ~1310 LSB)
            if peak >= 0.04:
                session.speech_started = True
                session.last_speech_time = time.perf_counter()

            # Apply 3x digital pre-scaling for Silero VAD neural net sensitivity
            vad_scaled = np.clip(vad_input * 3.0, -1.0, 1.0)
            result = session.vad_iterator(vad_scaled, return_seconds=False)
            if result is not None:
                if "start" in result:
                    session.speech_started = True
                    session.last_speech_time = time.perf_counter()
                    print("\n[vad] Speech started!")
                elif "end" in result:
                    await finalize_session(session, stop_callback, loop, reason="silence detected")
                    return

    # Post-speech Silence Cutoff: if speech occurred and room has been quiet for >= 1.0s, finalize immediately
    now = time.perf_counter()
    if session.speech_started and session.last_speech_time > 0 and (now - session.last_speech_time) >= 1.0:
        await finalize_session(session, stop_callback, loop, reason="post-speech silence (1.0s)")
        return

    # Fallback Timeout 1: No speech started within NO_SPEECH_TIMEOUT_SEC
    elapsed = now - session.wake_detected_time
    if not session.speech_started and elapsed >= NO_SPEECH_TIMEOUT_SEC:
        await finalize_session(session, stop_callback, loop, reason="no-speech timeout")
        return

    # Fallback Timeout 2: Hard max utterance duration
    if elapsed >= MAX_UTTERANCE_SEC:
        await finalize_session(session, stop_callback, loop, reason="max utterance duration")
        return


def update_telemetry(payload: dict):
    global latest_telemetry
    cpu = payload.get("cpu_percent", payload.get("cpu", 0))
    free_heap = payload.get("free_heap", payload.get("ram", 0))
    min_free = payload.get("min_free_heap", free_heap)
    used_heap = max(0, 262144 - free_heap)

    latest_telemetry.update({
        "cpu_percent": cpu,
        "cpu0_percent": payload.get("cpu0_percent", cpu),
        "cpu1_percent": payload.get("cpu1_percent", cpu),
        "free_heap": free_heap,
        "min_free_heap": min_free,
        "used_heap": used_heap,
        "uptime_ms": payload.get("uptime_ms", 0),
        "mic_peak": payload.get("mic_peak", 0),
        "raw_hex": payload.get("raw_hex", "0x00000000"),
        "last_updated": time.time(),
    })


# --------------------------------------------------------------------------
# WebSocket Transport (Wi-Fi Mode)
# --------------------------------------------------------------------------
@app.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[+] ESP32 connected via WebSocket (Wi-Fi).")
    loop = asyncio.get_running_loop()
    session = Session()

    async def send_stop():
        await websocket.send_text(json.dumps({"event": "stop"}))

    try:
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes"):
                await process_audio_chunk(session, msg["bytes"], send_stop, loop)

            elif msg.get("text"):
                try:
                    payload = json.loads(msg["text"])
                    event = payload.get("event")
                    if event == "start":
                        await process_start_event(session, source="Wi-Fi WS")
                    elif event in ("telemetry", "stats"):
                        update_telemetry(payload)
                        await broadcast_event({"type": "telemetry", "data": latest_telemetry})
                except Exception as e:
                    print(f"[warn] Error parsing WS message: {e}")
    except WebSocketDisconnect:
        pass
    print("[-] ESP32 disconnected from WebSocket.")


active_serial_conn = None


@app.post("/api/trigger")
@app.get("/api/trigger")
async def trigger_listening():
    global active_serial_conn
    if active_serial_conn and active_serial_conn.is_open:
        try:
            active_serial_conn.write(b"t\n")
            active_serial_conn.flush()
            return {"status": "ok", "message": "Trigger sent to ESP32"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": "ESP32 serial connection not active"}


# --------------------------------------------------------------------------
# Bulletproof USB Serial Transport (Demo Prototype Mode)
# --------------------------------------------------------------------------
async def serial_listener_task():
    global is_running, active_serial_conn
    try:
        import serial
    except ImportError:
        print("[Serial] pyserial not installed. Skipping direct USB Serial listener.")
        return

    while is_running:
        ser = None
        try:
            target_port = SERIAL_PORT
            try:
                import serial.tools.list_ports
                available_ports = [p.device for p in serial.tools.list_ports.comports()]
                if target_port not in available_ports and len(available_ports) > 0:
                    print(f"[Serial] {target_port} not found. Available COM ports on system: {available_ports}")
                    # Pick the first available port
                    target_port = available_ports[0]
                    print(f"[Serial] Auto-selecting {target_port}...")
            except Exception:
                pass

            print(f"[Serial] Connecting to ESP32 on {target_port} @ {SERIAL_BAUD} baud...")
            ser = serial.Serial(target_port, SERIAL_BAUD, timeout=0.05)
            active_serial_conn = ser
            print(f"[+] Connected to ESP32 on USB Serial ({target_port} @ {SERIAL_BAUD})!")

            session = Session()
            loop = asyncio.get_running_loop()

            async def send_serial_stop():
                try:
                    ser.write(b'{"event":"stop"}\n')
                    ser.flush()
                except Exception as e:
                    print(f"[Serial] Error writing stop: {e}")

            buffer = bytearray()

            while ser.is_open and is_running:
                raw = ser.read(2048)
                if raw:
                    buffer.extend(raw)

                    while len(buffer) >= 4:
                        idx = buffer.find(b'\xAA\x55')

                        if idx == -1:
                            if b'\n' in buffer:
                                line, _, remainder = buffer.partition(b'\n')
                                buffer = bytearray(remainder)
                                line_str = line.decode('utf-8', errors='ignore').strip()
                                if line_str.startswith('{') and line_str.endswith('}'):
                                    try:
                                        payload = json.loads(line_str)
                                        event = payload.get("event")
                                        if event == "start":
                                            await process_start_event(session, source="USB Serial")
                                        elif event in ("telemetry", "stats"):
                                            update_telemetry(payload)
                                            await broadcast_event({"type": "telemetry", "data": latest_telemetry})
                                    except Exception:
                                        pass
                            else:
                                if len(buffer) > 4096:
                                    del buffer[:2048]
                            break

                        elif idx > 0:
                            text_part = buffer[:idx]
                            del buffer[:idx]
                            for line in text_part.split(b'\n'):
                                line_str = line.decode('utf-8', errors='ignore').strip()
                                if line_str.startswith('{') and line_str.endswith('}'):
                                    try:
                                        payload = json.loads(line_str)
                                        event = payload.get("event")
                                        if event == "start":
                                            await process_start_event(session, source="USB Serial")
                                        elif event in ("telemetry", "stats"):
                                            update_telemetry(payload)
                                            await broadcast_event({"type": "telemetry", "data": latest_telemetry})
                                    except Exception:
                                        pass

                        else:
                            # idx == 0: 0xAA 0x55 binary frame
                            if len(buffer) < 4:
                                break
                            frame_len = (buffer[2] << 8) | buffer[3]

                            # Sanity check: valid frame length is 4 to 2048 bytes
                            if frame_len < 4 or frame_len > 2048:
                                del buffer[:2]  # Discard false header and resync
                                continue

                            if len(buffer) < 4 + frame_len:
                                break  # Wait for rest of frame

                            chunk = bytes(buffer[4:4 + frame_len])
                            del buffer[:4 + frame_len]
                            await process_audio_chunk(session, chunk, send_serial_stop, loop)

                # Timeout & silence watchdog
                if session.state == "STREAMING":
                    now = time.perf_counter()
                    elapsed = now - session.wake_detected_time
                    if (session.speech_started and session.last_speech_time > 0 and (now - session.last_speech_time) >= 1.0):
                        await finalize_session(session, send_serial_stop, loop, reason="post-speech silence")
                    elif (not session.speech_started and elapsed >= NO_SPEECH_TIMEOUT_SEC) or (elapsed >= MAX_UTTERANCE_SEC):
                        await finalize_session(session, send_serial_stop, loop, reason="timeout")

                await asyncio.sleep(0.005)

        except Exception as ex:
            if ser and ser.is_open:
                try: ser.close()
                except Exception: pass
            await asyncio.sleep(1.5)


@app.on_event("startup")
async def startup_tasks():
    asyncio.create_task(serial_listener_task())


@app.on_event("shutdown")
async def shutdown_tasks():
    global is_running
    is_running = False


# --------------------------------------------------------------------------
# Dashboard & REST API
# --------------------------------------------------------------------------
@app.get("/events")
async def sse_events(request: Request):
    async def event_generator():
        q = asyncio.Queue()
        event_subscribers.append(q)
        try:
            yield f"data: {json.dumps({'type': 'telemetry', 'data': latest_telemetry})}\n\n"
            yield f"data: {json.dumps({'type': 'state', 'state': latest_metrics['state']})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                data = await q.get()
                yield data
        finally:
            if q in event_subscribers:
                event_subscribers.remove(q)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/status")
async def get_status():
    return {
        "metrics": latest_metrics,
        "telemetry": latest_telemetry,
        "history": transcript_history[-10:],
    }


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SIH Edge AI - ESP32-S3 Voice Assistant Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @keyframes pulse-slow { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .pulse-dot { animation: pulse-slow 1.5s infinite ease-in-out; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 font-sans min-h-screen p-6">
  <div class="max-w-6xl mx-auto space-y-6">
    <header class="flex flex-col md:flex-row items-start md:items-center justify-between border-b border-slate-800 pb-4">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
          <span class="text-cyan-400">🎙️ Smart India Hackathon</span>
          <span class="text-sm bg-cyan-950/80 border border-cyan-700/50 text-cyan-300 px-2.5 py-0.5 rounded-full">Edge AI Pipeline</span>
        </h1>
        <p class="text-xs text-slate-400 mt-1">ESP32-S3 (On-device KWS) ──> USB Serial (921600) ──> Faster-Whisper + Silero VAD</p>
      </div>
      <div class="flex flex-wrap items-center gap-3">
        <!-- Live Diagnostic Mic Status LED -->
        <div id="micLedBadge" class="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold bg-red-950/80 border border-red-600 text-red-300 transition-all">
          <span id="micLedDot" class="w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span>
          <span id="micLedText">🔴 MIC: WAITING FOR SIGNAL</span>
        </div>

        <!-- Pipeline Status Badge -->
        <div id="statusBadge" class="flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold bg-emerald-950/80 border border-emerald-600 text-emerald-300">
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 pulse-dot"></span>
          <span id="statusText">IDLE (WAITING FOR WAKE WORD)</span>
        </div>

        <!-- Manual Trigger Button -->
        <button onclick="triggerESP()" class="px-3.5 py-1.5 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-bold transition flex items-center gap-1.5 shadow-md active:scale-95">
          <span>🎙️ Test Trigger (BOOT)</span>
        </button>
      </div>
    </header>

    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
        <div class="flex items-center justify-between text-xs text-slate-400">
          <span>ESP32-S3 CPU Load</span>
          <span class="px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px] font-mono">&lt;10% Target</span>
        </div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="cpuVal" class="text-3xl font-extrabold text-cyan-400">0</span>
          <span class="text-sm text-slate-400">%</span>
        </div>
        <div class="w-full bg-slate-800 rounded-full h-2 mt-3 overflow-hidden">
          <div id="cpuBar" class="bg-cyan-400 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
        <div class="flex items-center justify-between text-xs text-slate-400">
          <span>RAM Budget (256 KB)</span>
          <span id="ramFree" class="text-[10px] font-mono text-emerald-400">256 KB Free</span>
        </div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="ramUsed" class="text-3xl font-extrabold text-purple-400">0</span>
          <span class="text-sm text-slate-400">KB Used</span>
        </div>
        <div class="w-full bg-slate-800 rounded-full h-2 mt-3 overflow-hidden">
          <div id="ramBar" class="bg-purple-500 h-2 rounded-full transition-all duration-300" style="width: 0%"></div>
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
        <div class="text-xs text-slate-400">Whisper ASR Latency</div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="whisperLat" class="text-3xl font-extrabold text-amber-400">0</span>
          <span class="text-sm text-slate-400">ms</span>
        </div>
        <p class="text-[11px] text-slate-400 mt-3">Faster-Whisper (int8 CPU)</p>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
        <div class="text-xs text-slate-400">Total Pipeline Latency</div>
        <div class="mt-2 flex items-baseline gap-2">
          <span id="totalLat" class="text-3xl font-extrabold text-emerald-400">0</span>
          <span class="text-sm text-slate-400">ms</span>
        </div>
        <p id="audioLen" class="text-[11px] text-slate-400 mt-3">Wake ➔ Speech ➔ Final Text</p>
      </div>
    </div>

    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-sm font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
          <span>⚡ Live Transcription Result</span>
        </h2>
        <span id="lastTime" class="text-xs text-slate-400">Waiting for utterance...</span>
      </div>
      <div id="latestTranscriptBox" class="bg-slate-950 border border-slate-800 rounded-lg p-4 min-h-[90px] flex items-center">
        <p id="latestTranscript" class="text-xl font-medium text-slate-100 italic">Say the wake word and speak your command...</p>
      </div>
    </div>

    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <h2 class="text-sm font-semibold text-slate-300 uppercase tracking-wider">Session History Log</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs text-slate-300">
          <thead class="text-[11px] uppercase bg-slate-800/60 text-slate-400">
            <tr>
              <th class="p-3">Time</th>
              <th class="p-3">Transcription</th>
              <th class="p-3">Audio Duration</th>
              <th class="p-3">Whisper Latency</th>
              <th class="p-3">Total E2E</th>
            </tr>
          </thead>
          <tbody id="historyBody" class="divide-y divide-slate-800">
            <tr><td colspan="5" class="p-4 text-center text-slate-400">No utterances recorded yet.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const evtSource = new EventSource('/events');

    evtSource.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type === 'telemetry') {
        updateTelemetry(msg.data);
      } else if (msg.type === 'state') {
        updateState(msg.state);
      } else if (msg.type === 'result') {
        addResult(msg.data);
      }
    };

    let latestPipelineStatus = 'IDLE';

    function updateState(state) {
      latestPipelineStatus = state;
      const badge = document.getElementById('statusBadge');
      const text = document.getElementById('statusText');
      const ledBadge = document.getElementById('micLedBadge');
      const ledDot = document.getElementById('micLedDot');
      const ledText = document.getElementById('micLedText');

      if (state === 'STREAMING') {
        badge.className = 'flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold bg-cyan-950/80 border border-cyan-500 text-cyan-300';
        text.innerText = '🎙️ LISTENING / STREAMING AUDIO';
        // Flash Red on trigger / streaming
        ledBadge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold bg-red-950/90 border border-red-500 text-red-200 animate-pulse';
        ledDot.className = 'w-3 h-3 rounded-full bg-red-500 shadow-[0_0_12px_rgba(239,68,68,1)]';
        ledText.innerText = '🚨 WAKE WORD TRIGGERED / STREAMING';
      } else if (state === 'TRANSCRIBING') {
        badge.className = 'flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold bg-amber-950/80 border border-amber-500 text-amber-300';
        text.innerText = '⚡ RUNNING WHISPER ASR';
      } else {
        badge.className = 'flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold bg-emerald-950/80 border border-emerald-600 text-emerald-300';
        text.innerText = 'IDLE (WAITING FOR WAKE WORD)';
      }
    }

    function updateTelemetry(t) {
      document.getElementById('cpuVal').innerText = t.cpu_percent || 0;
      document.getElementById('cpuBar').style.width = Math.min(t.cpu_percent || 0, 100) + '%';

      const usedKb = Math.round((t.used_heap || 0) / 1024);
      const freeKb = Math.round((t.free_heap || 0) / 1024);
      document.getElementById('ramUsed').innerText = usedKb;
      document.getElementById('ramFree').innerText = freeKb + ' KB Free';
      document.getElementById('ramBar').style.width = Math.min((usedKb / 256) * 100, 100) + '%';

      const peak = t.mic_peak || 0;
      const rawHex = t.raw_hex || "0x00000000";
      const ledBadge = document.getElementById('micLedBadge');
      const ledDot = document.getElementById('micLedDot');
      const ledText = document.getElementById('micLedText');

      if (latestPipelineStatus !== 'STREAMING') {
        if (peak >= 500) {
          ledBadge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold bg-emerald-950/80 border border-emerald-500 text-emerald-300';
          ledDot.className = 'w-3 h-3 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.9)]';
          ledText.innerText = '🟢 MIC DETECTING AUDIO (Peak: ' + peak + ')';
        } else {
          ledBadge.className = 'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold bg-red-950/80 border border-red-600 text-red-300';
          ledDot.className = 'w-3 h-3 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]';
          ledText.innerText = '🔴 CONSTANT RED: SILENT / WAITING (Peak: ' + peak + ')';
        }
      }
    }

    async function triggerESP() {
      try {
        const res = await fetch('/api/trigger', { method: 'POST' });
        const data = await res.json();
        console.log('Trigger result:', data);
      } catch (err) {
        console.error('Trigger error:', err);
      }
    }

    function addResult(r) {
      document.getElementById('whisperLat').innerText = r.whisper_latency_ms;
      document.getElementById('totalLat').innerText = r.total_e2e_latency_ms;
      document.getElementById('audioLen').innerText = 'Audio length: ' + r.audio_duration_sec + 's';
      document.getElementById('latestTranscript').innerText = '\"' + r.transcript + '\"';
      document.getElementById('latestTranscript').className = 'text-xl font-medium text-emerald-300';
      document.getElementById('lastTime').innerText = new Date(r.timestamp * 1000).toLocaleTimeString();

      const tbody = document.getElementById('historyBody');
      const emptyRow = tbody.querySelector('tr td[colspan]');
      if (emptyRow) tbody.innerHTML = '';

      const row = `
        <tr class="hover:bg-slate-800/40 transition">
          <td class="p-3 text-slate-400">${new Date(r.timestamp * 1000).toLocaleTimeString()}</td>
          <td class="p-3 font-medium text-white">${r.transcript}</td>
          <td class="p-3 text-slate-400">${r.audio_duration_sec}s</td>
          <td class="p-3 text-amber-400 font-mono">${r.whisper_latency_ms} ms</td>
          <td class="p-3 text-emerald-400 font-mono font-semibold">${r.total_e2e_latency_ms} ms</td>
        </tr>
      `;
      tbody.innerHTML = row + tbody.innerHTML;
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    try:
        uvicorn.run(app, host=HOST, port=PORT)
    except KeyboardInterrupt:
        print("\n[Server] Exited.")
        os._exit(0)
