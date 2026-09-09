"""
Unified Server for Voice-Activated Edge Device (ESP32-S3 + KWS + Whisper).
Supports both:
  1. USB Serial (/dev/ttyACM* or COM7 @ 921600 baud) for prototype demo.
  2. Wi-Fi WebSockets (ws://host:8080/stream) for final deployment.
"""

import asyncio
import base64
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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


HALLUCINATIONS = {
    "you", "you.", "you!", "you you", "you, you", "you you you",
    "thank you.", "thank you", "thank you!", "thank you for watching.",
    "thank you for watching", "subtitles by", "subtitles",
    "subtitles by the amara.org community", "bye", "bye.", "bye!"
}


def transcribe_sync(audio_bytes: bytes) -> str:
    audio_np = pcm_bytes_to_float32(audio_bytes)
    if len(audio_np) == 0:
        return ""

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    max_amp = float(np.max(np.abs(audio_np)))

    # Reject if audio is ambient noise or low-amplitude room hiss (< 0.035 peak or < 0.005 RMS)
    if max_amp < 0.035 and rms < 0.005:
        print(f"[whisper] Rejecting ambient noise (max_amp={max_amp:.4f}, rms={rms:.4f})")
        return ""

    # Peak Normalization: only for real speech signals (max_amp >= 0.04)
    if 0.04 <= max_amp < 0.70:
        gain = min(2.5, 0.70 / max_amp)
        audio_np = audio_np * gain

    # Whisper transcription with strict anti-hallucination settings
    segments, _ = whisper_model.transcribe(
        audio_np,
        language="en",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, threshold=0.5, speech_pad_ms=80),
        condition_on_previous_text=False,
        no_speech_threshold=0.55,
        compression_ratio_threshold=2.0,
        repetition_penalty=1.25,
        hallucination_silence_threshold=0.5,
    )
    res = " ".join(seg.text.strip() for seg in segments).strip()

    # Suppress common Whisper hallucination loops on ambient noise
    cleaned_lower = res.lower().strip().rstrip(".!?,")
    if cleaned_lower in HALLUCINATIONS or len(cleaned_lower) <= 1:
        print(f"[whisper] Suppressed hallucination: {res!r}")
        return ""

    # Check for repetitive loops like "you, you, you"
    words = cleaned_lower.split()
    if len(words) >= 3 and len(set(words)) == 1:
        print(f"[whisper] Suppressed repetitive loop: {res!r}")
        return ""

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

    if duration_sec < 0.4 or (not session.speech_started and session.max_peak < 0.04):
        print(f"[transcript] (Ambient noise / no speech detected, skipping Whisper: peak={session.max_peak:.3f})")
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

    # Broadcast audio chunk to web dashboard for live Spectrogram & Oscilloscope!
    try:
        b64_pcm = base64.b64encode(chunk).decode('ascii')
        await broadcast_event({"type": "audio_chunk", "pcm16_b64": b64_pcm})
    except Exception:
        pass

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
    cpu0 = payload.get("cpu0_percent", cpu)
    cpu1 = payload.get("cpu1_percent", cpu)
    free_heap = payload.get("free_heap", payload.get("ram", 0))
    min_free = payload.get("min_free_heap", free_heap)
    used_heap = max(0, 327680 - free_heap)
    gatekeeper = payload.get("gatekeeper", payload.get("acoustic_state", latest_telemetry.get("gatekeeper", "SILENCE")))
    transport = payload.get("transport", latest_telemetry.get("transport", "Waiting..."))

    cpu_status = "OPTIMAL" if cpu < 75 else "HIGH LOAD"
    ram_status = "OPTIMAL" if free_heap > 65536 else "LOW MEMORY"

    latest_telemetry.update({
        "cpu_percent": cpu,
        "cpu0_percent": cpu0,
        "cpu1_percent": cpu1,
        "free_heap": free_heap,
        "min_free_heap": min_free,
        "used_heap": used_heap,
        "uptime_ms": payload.get("uptime_ms", 0),
        "mic_peak": payload.get("mic_peak", 0),
        "raw_hex": payload.get("raw_hex", "0x00000000"),
        "gatekeeper": gatekeeper,
        "transport": transport,
        "cpu_status": cpu_status,
        "ram_status": ram_status,
        "last_updated": time.time(),
    })


active_serial_conn = None
active_ws_conn = None

# --------------------------------------------------------------------------
# WebSocket Transport (Wi-Fi Mode)
# --------------------------------------------------------------------------
@app.websocket("/stream")
@app.websocket("/ws")
@app.websocket("/ws/audio")
async def stream_endpoint(websocket: WebSocket):
    global active_ws_conn
    await websocket.accept()
    active_ws_conn = websocket
    latest_telemetry["transport"] = "Wi-Fi WebSocket"
    await broadcast_event({"type": "transport", "transport": "Wi-Fi WebSocket"})
    print("[+] ESP32 connected via WebSocket (Wi-Fi).")
    loop = asyncio.get_running_loop()
    session = Session()

    async def send_stop():
        try:
            await websocket.send_text(json.dumps({"event": "stop"}))
        except Exception:
            pass

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
                        payload["transport"] = "Wi-Fi WebSocket"
                        update_telemetry(payload)
                        await broadcast_event({"type": "telemetry", "data": latest_telemetry})
                except Exception as e:
                    print(f"[warn] Error parsing WS message: {e}")
    except WebSocketDisconnect:
        pass
    finally:
        active_ws_conn = None
        print("[-] ESP32 disconnected from WebSocket.")


@app.post("/api/trigger")
@app.get("/api/trigger")
async def trigger_listening():
    global active_serial_conn, active_ws_conn
    sent = False
    if active_ws_conn:
        try:
            await active_ws_conn.send_text(json.dumps({"event": "trigger"}))
            sent = True
        except Exception as e:
            print(f"[trigger] WS send error: {e}")
    if active_serial_conn and active_serial_conn.is_open:
        try:
            active_serial_conn.write(b"t\n")
            active_serial_conn.flush()
            sent = True
        except Exception as e:
            print(f"[trigger] Serial send error: {e}")
    if sent:
        return {"status": "ok", "message": "Trigger sent to ESP32 (Wi-Fi/Serial)"}
    return {"status": "error", "message": "Neither Wi-Fi nor Serial ESP32 connection is active"}


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
            target_port = os.environ.get("ESP32_SERIAL_PORT", SERIAL_PORT)
            try:
                import serial.tools.list_ports
                ports = list(serial.tools.list_ports.comports())
                port_devices = [p.device for p in ports]
                if target_port not in port_devices and len(ports) > 0:
                    # Prefer known USB-to-UART bridge chips (CP210x, CH340, FTDI, Espressif)
                    esp_ports = [
                        p.device for p in ports 
                        if any(k in (f"{p.description} {getattr(p, 'hwid', '')}").upper() 
                               for k in ["CP210", "CH34", "CH91", "UART", "ESPRESSIF", "303A", "USB SERIAL"])
                    ]
                    if esp_ports:
                        target_port = esp_ports[0]
                    else:
                        target_port = ports[-1].device
                    print(f"[Serial] Auto-selected ESP32 port: {target_port} (Detected ports: {port_devices})")
            except Exception:
                pass

            print(f"[Serial] Connecting to ESP32 on {target_port} @ {SERIAL_BAUD} baud...")
            ser = serial.Serial(target_port, SERIAL_BAUD, timeout=0.01)
            active_serial_conn = ser
            transport_label = f"USB Serial ({target_port})"
            latest_telemetry["transport"] = transport_label
            await broadcast_event({"type": "transport", "transport": transport_label})
            await broadcast_event({"type": "telemetry", "data": latest_telemetry})
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
                raw = None
                try:
                    waiting = ser.in_waiting
                    if waiting > 0:
                        raw = ser.read(min(waiting, 4096))
                    else:
                        await asyncio.sleep(0.005)
                except Exception as read_err:
                    print(f"[Serial] Port read error: {read_err}")
                    break

                if raw:
                    buffer.extend(raw)

                while len(buffer) >= 4:
                    idx = buffer.find(b'\xAA\x55')

                    if idx == -1:
                        while b'\n' in buffer:
                            line, _, remainder = buffer.partition(b'\n')
                            buffer = bytearray(remainder)
                            line_str = line.decode('utf-8', errors='ignore').strip()
                            if line_str:
                                print(f"[Serial RX] {line_str}")
                            if line_str.startswith('{') and line_str.endswith('}'):
                                try:
                                    payload = json.loads(line_str)
                                    event = payload.get("event")
                                    if event == "start":
                                        await process_start_event(session, source=transport_label)
                                    elif event in ("telemetry", "stats"):
                                        payload["transport"] = transport_label
                                        update_telemetry(payload)
                                        await broadcast_event({"type": "telemetry", "data": latest_telemetry})
                                except Exception as e:
                                    print(f"[Serial] JSON parse error: {e}")
                        if len(buffer) > 4096:
                            del buffer[:2048]
                        break

                    elif idx > 0:
                        text_part = buffer[:idx]
                        del buffer[:idx]
                        for line in text_part.split(b'\n'):
                            line_str = line.decode('utf-8', errors='ignore').strip()
                            if line_str:
                                print(f"[Serial RX] {line_str}")
                            if line_str.startswith('{') and line_str.endswith('}'):
                                try:
                                    payload = json.loads(line_str)
                                    event = payload.get("event")
                                    if event == "start":
                                        await process_start_event(session, source=transport_label)
                                    elif event in ("telemetry", "stats"):
                                        payload["transport"] = transport_label
                                        update_telemetry(payload)
                                        await broadcast_event({"type": "telemetry", "data": latest_telemetry})
                                except Exception as e:
                                    print(f"[Serial] JSON parse error: {e}")

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

                await asyncio.sleep(0.002)

        except Exception as ex:
            print(f"[Serial] Connection error: {ex}")
            active_serial_conn = None
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


DASHBOARD_FILE = os.path.join(os.path.dirname(__file__), "dashboard.html")


@app.get("/dashboard")
@app.get("/")
async def dashboard():
    if os.path.exists(DASHBOARD_FILE):
        return FileResponse(DASHBOARD_FILE)
    return HTMLResponse("<h1>Audi's Dashboard file not found at " + DASHBOARD_FILE + "</h1>")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESP32-S3 Speech AI Server")
    parser.add_argument("--port", type=str, default=SERIAL_PORT, help="ESP32 COM port (e.g. COM7 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=SERIAL_BAUD, help="ESP32 Serial Baud Rate (default: 921600)")
    parser.add_argument("--host", type=str, default=HOST, help="HTTP Server Host (default: 0.0.0.0)")
    parser.add_argument("--http-port", type=int, default=PORT, help="HTTP Server Port (default: 8080)")
    args, _ = parser.parse_known_args()

    SERIAL_PORT = args.port
    SERIAL_BAUD = args.baud
    HOST = args.host
    PORT = args.http_port

    try:
        import serial.tools.list_ports
        available = [f"{p.device} ({p.description})" for p in serial.tools.list_ports.comports()]
        print(f"[System] Available Serial Devices: {available if available else 'None found'}")
    except Exception:
        pass

    print(f"[Server] Starting FastAPI Server on http://{HOST}:{PORT}")
    print(f"[Server] Dashboard URL: http://localhost:{PORT}")
    print(f"[Server] Configured Serial Target: {SERIAL_PORT} @ {SERIAL_BAUD} baud")

    try:
        uvicorn.run(app, host=HOST, port=PORT)
    except KeyboardInterrupt:
        print("\n[Server] Exited.")
        os._exit(0)
