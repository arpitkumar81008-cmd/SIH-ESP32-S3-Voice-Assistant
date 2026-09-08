# Edge-to-Cloud Voice Assistant: Setup & Deployment Guide

This document provides complete, end-to-end instructions for configuring the **ESP32-S3 Edge Voice Assistant** hardware firmware, running on-device **Keyword Spotting (KWS)** with the **TENet INT8 Model** and **Acoustic Gatekeeper**, and streaming live audio to the **FastAPI + Faster-Whisper Server** and **Cyber-Telemetry Dashboard**.

---

## 1. System Architecture

```
                       ESP32-S3 Edge Node (Core 0 & Core 1)
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                                                                            │
  │  [INMP441 Mic] ──► [I2S DMA Rx] ──► [DC High-Pass] ──► [Gain Scaling >>11]│
  │                                                                │           │
  │                                                                ▼           │
  │                                                    [True FIFO Ring Buffer] │
  │                                                                │           │
  │             ┌──────────────────────────────────────────────────┤           │
  │             ▼                                                  ▼           │
  │   [Acoustic Gatekeeper]                              [Stream Manager]      │
  │    * Silence Detection (peak < 550)                   * 500ms Lookback     │
  │    * Ambient Noise Rejection (0-250 Hz)               * Zero-Drop Ring FIFO│
  │    * Voice Band Energy (300-3400 Hz)                  * Binary PCM Output  │
  │             │                                                  │           │
  │             ▼ (Voice Detected)                                 │           │
  │   [TENet TFLM Engine]                                          │           │
  │    * 1.0s Window / 200ms Hop                                   │           │
  │    * 51x10 MFCC Spectrogram                                    │           │
  │    * INT8 Model ("Ankit" Trigger) ───► [Trigger Queue] ────────┘           │
  │                                                                            │
  └──────────────────────────────────────┬─────────────────────────────────────┘
                                         │
                 USB Serial (921,600 baud) OR Wi-Fi WebSocket (ws://ip:8080)
                                         │
                                         ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │                    FastAPI Backend & Ingestion Engine                      │
  │  * Silero VAD (Dynamic Speech Endpointing)                                 │
  │  * Faster-Whisper ASR (base.en / int8 quantization)                       │
  │  * Keyword Spotting Server-Side Verification ("Ankit")                     │
  │  * Audi-Themed Cyber-Telemetry & Live Audio Lab Dashboard (SSE Broadcast)   │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Wiring & Pinout

The system uses an **ESP32-S3 Dev Module** (Dual-Core Xtensa LX7 @ 240 MHz) paired with an **INMP441 MEMS Omnidirectional Microphone**.

| INMP441 Pin | ESP32-S3 Pin | Function / Description |
|---|---|---|
| **WS** (Word Select) | **GPIO 4** | I2S Word Select / Left-Right Clock (LRCLK) |
| **SCK** (Clock) | **GPIO 5** | Continuous Serial Bit Clock (BCLK) |
| **SD** (Serial Data) | **GPIO 7** | Serial Audio Data Output from Microphone |
| **VDD** | **3.3V** | Regulated 3.3V DC Power (Do NOT connect to 5V) |
| **GND** | **GND** | Ground Reference |
| **L/R** | **GND** | Ties channel selection to Left slot (Standard mono) |
| **BOOT Button** | **GPIO 0** | Physical hardware trigger backup |
| **RGB LED** | **GPIO 48** | Onboard WS2812 Addressable RGB LED (or GPIO 2 standard LED) |

> [!IMPORTANT]
> **SD Pull-Down**: The firmware configures an internal pull-down resistor on GPIO 7 (`gpio_set_pull_mode(I2S_SD_PIN, GPIO_PULLDOWN_ONLY)`). This prevents floating lines from producing `0xFFFFFFFF` artifacts during silence.

---

## 3. Software & Toolchain Prerequisites

### Arduino IDE Setup
1. Download and install **Arduino IDE 2.x** (or Arduino CLI).
2. Open **File → Preferences** and add the ESP32 board manager URL:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Open **Tools → Board → Boards Manager**, search for `esp32` by Espressif, and install version **2.0.14** or **3.x**.

### Required Arduino Libraries
Open **Tools → Manage Libraries...** and install the following:
1. **`tflm_esp32`** (v2.0.0 by Simone Salerno / EloquentArduino): Precompiled TensorFlow Lite Micro binaries for ESP32-S3.
2. **`arduinoFFT`** (v2.0.4 by Enrique Condes / Bim Overbohm): Fast Fourier Transform library supporting templated single-precision `float`.
3. **`WebSockets`** (by Markus Sattler): Lightweight client/server WebSocket transport (required if `ENABLE_NETWORK_STREAM = 1`).
4. **`ArduinoJson`** (v6.x or v7.x by Benoit Blanchon): High-speed JSON serialization for telemetry and events.

---

## 4. Arduino IDE Board Configuration

Under the **Tools** menu in Arduino IDE, ensure the following parameters are selected:

* **Board:** `ESP32S3 Dev Module`
* **USB CDC On Boot:** `Enabled` *(Crucial for Serial Monitor and 921600 baud streaming)*
* **CPU Frequency:** `240MHz (WiFi)`
* **Flash Mode:** `QIO 80MHz` (or `OPI 80MHz` if using WROOM-2 with Octal Flash)
* **Flash Size:** `8MB (64Mb)` or `16MB (128Mb)`
* **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)` or `Default 4MB with spiffs`
* **PSRAM:** `OPI PSRAM` (or `Enabled`)
* **Core Debug Level:** `None` (or `Info` for debugging)
* **Port:** Select your ESP32-S3 COM port.

---

## 5. Firmware Configuration & Operating Modes

Open `firmware/esp32_sih_hard_ware_firmware.ino` and review the build flags:

```cpp
// ---------------- BUILD FLAGS ----------------
#define ENABLE_NETWORK_STREAM 0   // 0 = USB Serial (Demo), 1 = Wi-Fi WebSocket
#define USE_FAKE_AUDIO        0   // 0 = Real INMP441 I2S mic, 1 = Synth sine wave
#define USE_PHYSICAL_BUTTON   1   // 1 = BOOT button (GPIO 0) backup trigger
#define USE_STATUS_LED        1   // 1 = Status LED active
```

### Mode 0: USB Serial (Plug-and-Play Demo)
* Streams framed 16-bit binary PCM chunks directly over USB CDC at **921,600 baud**.
* Emits real-time telemetry JSON strings every 3 seconds.
* Zero network setup required; ideal for evaluation and offline demonstrations.

### Mode 1: Wi-Fi WebSocket (IoT Network Deployment)
* Set `#define ENABLE_NETWORK_STREAM 1`.
* Enter your 2.4 GHz Wi-Fi credentials and PC IPv4 address:
  ```cpp
  #define WIFI_SSID   "Your-2.4GHz-SSID"
  #define WIFI_PASS   "Your-Password"
  #define WS_HOST     "192.168.1.100"   // PC IPv4 address from ipconfig / ifconfig
  #define WS_PORT     8080
  #define WS_PATH     "/stream"
  ```

---

## 6. Acoustic Gatekeeper & Voice Thresholding

The firmware incorporates a **two-tier acoustic gatekeeper** running on Core 0:

```
[Raw Audio Chunk (512 samples / 32ms)]
                │
                ▼
       [Peak Energy < 550?] ────────► [ACOUSTIC_SILENCE] ──► Constant RED LED
                │                                             (TFLM Bypassed)
                ▼ (Sound Present)
  [512-pt FFT Spectral Analysis]
    * Low-Freq Noise (0-250 Hz)
    * Speech Formants (300-3400 Hz)
                │
   [Voice Band < 3500.0f OR ────────► [ACOUSTIC_ROOM_AUDIO] ► Dim BLUE LED
    Noise Dominates Speech?]                                   (TFLM Bypassed)
                │
                ▼ (Speech Formants Dominant)
      [ACOUSTIC_VOICE] ─────────────► Solid GREEN LED
                │
                ▼
   [Gatekeeper OPEN: Invoke TENet KWS on 1.0s Sliding Window]
```

### Visual LED Status Guide
| LED Color / State | System Status | Acoustic Gatekeeper |
|---|---|---|
| **Constant RED** | Room Silence / Idle | **CLOSED** (Model bypassed, CPU < 5%) |
| **Dim BLUE** | Ambient Room Noise / Fan Hum | **CLOSED** (Noise rejected, model bypassed) |
| **Solid GREEN** | Active Voice Detected | **OPEN** (TENet KWS evaluating speech) |
| **4x Red Flashes** | Wake Word ("Ankit") Detected! | **TRIGGERED** (Lookback initiated) |
| **Solid GREEN** | Active Audio Streaming | Audio streaming to Whisper server |

---

## 7. TENet KWS Neural Network Model

The embedded Keyword Spotting model is a **TENet Inverted Residual Convolutional Neural Network**:
* **Header File:** `firmware/kws_model_data.h` (`g_kws_model_data[]`, 57,264 bytes)
* **Input Tensor:** `[1, 51, 1, 10]` `INT8` (51 temporal frames $\times$ 10 spectral bins)
* **Output Tensor:** `[1, 5]` `INT8` (Softmax probability over 5 classes)
* **Classes:**
  - `Index 0`: **`wake_word` ("Ankit")**
  - `Index 1`: `local_negative` (phonetically similar confuser words)
  - `Index 2`: `noise` (ambient acoustic disturbances)
  - `Index 3`: `silence` (dead air)
  - `Index 4`: `unknown` (other conversational speech)
* **Trigger Condition:** Raw INT8 score $\ge 0$ (corresponding to $\ge 50\%$ probability) with wake score strictly greater than negative and noise classes.

---

## 8. Python Server & Cyber-Telemetry Dashboard

### 1. Python Environment Setup
```bash
cd SIH-ESP32-S3-Voice-Assistant/server

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install required packages
pip install fastapi uvicorn websockets pyserial numpy soundfile torch faster-whisper
```

### 2. Launching the Server
```bash
# Start unified ingestion engine and web server
python server.py
```
The server will bind to port `8080` and log initialization:
```
[init] Loading Silero VAD model...
[init] Loading Faster-Whisper model 'base.en' (device=cpu, compute_type=int8)...
[server] FastAPI Server listening on http://0.0.0.0:8080
```

### 3. Accessing the Dashboard
Open your browser to:
```
http://localhost:8080
```
or from any device on your local network:
```
http://<PC-IP-ADDRESS>:8080
```

The **Audi-Themed Cyber-Telemetry Dashboard** displays:
* Real-time circular VU level meter and live audio oscilloscope
* Real-time 2D multi-color voice spectrogram
* FreeRTOS CPU load (Core 0, Core 1, and Average) & internal SRAM heap monitor
* Acoustic Gatekeeper live status (`SILENCE` / `ROOM_AUDIO` / `VOICE`)
* Live Whisper transcription history with verified Keyword Spotting badges

---

## 9. Troubleshooting & Diagnostics

### Microphone Reads `0xFFFFFFFF` or Flat 0
* **Cause:** The SD (Serial Data) pin is disconnected or floating.
* **Fix:** Verify jumper wire connection to **GPIO 7**. Ensure `L/R` and `GND` pins on the INMP441 breakout are both firmly grounded.

### Serial Monitor Displays Garbled Characters
* **Cause:** Baud rate mismatch.
* **Fix:** Set Serial Monitor baud rate to **921600 baud**. Make sure **USB CDC On Boot** is set to **Enabled** in Tools.

### Wi-Fi Connection Fails
* **Cause:** ESP32-S3 supports 2.4 GHz Wi-Fi only.
* **Fix:** Ensure your Wi-Fi router broadcasts a separate 2.4 GHz network (5 GHz is unsupported).

### False Triggering on Room Noise
* **Cause:** Acoustic Gatekeeper threshold is too low for your room acoustics.
* **Fix:** In `firmware/esp32_sih_hard_ware_firmware.ino`, increase `SILENCE_ENERGY_THRESHOLD` (e.g. from 550 to 750) or increase `VOICE_BAND_ENERGY_THRESHOLD` (e.g. from 3500 to 5000).
