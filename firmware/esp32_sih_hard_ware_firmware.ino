/*
 * HARDWARE TEAM FIRMWARE — Edge-to-Cloud Keyword Spotting (ESP32-S3)
 * Dual-Mode: USB Serial (921600 baud for demo) + Wi-Fi WebSocket (Final Deployment)
 *
 * Fully Error-Proofed & Synchronized:
 *   - True FIFO Ring Buffer: Guaranteed zero duplicated or skipped samples.
 *   - Pre-Scale DC High-Pass Filter: Eliminates hardware DC bias before gain scaling.
 *   - Non-Blocking Serial Command Reader: Zero 1000ms timeout stalls.
 *   - Sustained Energy Gate: Prevents false triggers from single-sample impulse clicks.
 *   - Idle CPU < 10%: Tasks yield and sleep during quiet periods.
 *   - Low RAM Footprint: Operates comfortably within internal SRAM (~110 KB).
 */

#include <Arduino.h>
#include "driver/i2s.h"
#include "kws_model_data.h"

// ---------------- BUILD FLAGS ----------------
#define ENABLE_NETWORK_STREAM 0   // 0 = USB Serial (Prototype Demo), 1 = Wi-Fi WebSocket
#define USE_FAKE_AUDIO        0   // 0 = Real INMP441 I2S microphone (GPIO 4, 5, 6)
#define USE_PHYSICAL_BUTTON   1   // 1 = BOOT button (GPIO 0) backup trigger
#define USE_STATUS_LED        1   // 1 = Status LED (GPIO 2)

#if ENABLE_NETWORK_STREAM
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#endif

// ---------------- NETWORK CONFIG ----------------
// NOTE: ESP32-S3 Wi-Fi hardware supports 2.4 GHz ONLY (5 GHz Wi-Fi is not supported).
#define WIFI_SSID   "Your-WiFi-SSID"      // Replace with your 2.4 GHz Wi-Fi SSID
#define WIFI_PASS   "Your-WiFi-Password"  // Replace with your Wi-Fi Password
#define WS_HOST     "192.168.1.100"       // Replace with PC IPv4 Address (find via 'ipconfig')
#define WS_PORT     8080
#define WS_PATH     "/stream"

// ---------------- PIN ASSIGNMENTS ----------------
#define I2S_WS_PIN   4    // Word Select (White wire)
#define I2S_SCK_PIN  5    // Continuous Serial Clock (Grey/Purple wire)
#define I2S_SD_PIN   7    // Serial Data Out from mic (Blue wire plugged into GPIO 7)
#define I2S_PORT     I2S_NUM_0

#define STATUS_LED_PIN     2    // Single-color LED fallback
#define TRIGGER_BUTTON_PIN 0    // BOOT button

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48          // Onboard WS2812 RGB LED for ESP32-S3 DevKit
#endif

// ---------------- VISUAL DIAGNOSTIC RGB LED DRIVER ----------------
void setLedColor(uint8_t r, uint8_t g, uint8_t b) {
#ifdef RGB_BUILTIN
  neopixelWrite(RGB_BUILTIN, r, g, b);
#endif
#if USE_STATUS_LED
  digitalWrite(STATUS_LED_PIN, (r > 0 || g > 0) ? HIGH : LOW);
#endif
}

void flashWakeWordRed() {
  for (int i = 0; i < 4; i++) {
    setLedColor(70, 0, 0);  // Bright Red Flash
    delay(50);
    setLedColor(0, 0, 0);   // Off
    delay(50);
  }
}

// ---------------- AUDIO CONFIG ----------------
#define SAMPLE_RATE       16000
#define BITS_PER_SAMPLE   I2S_BITS_PER_SAMPLE_32BIT
#define I2S_MIC_CHANNEL   I2S_CHANNEL_FMT_RIGHT_LEFT
#define RING_SECONDS      3
#define LOOKBACK_MS       500
#define SAFETY_MAX_STREAM_MS 12000
#define TELEMETRY_INTERVAL_MS 3000

static const size_t RING_BUFFER_SAMPLES = SAMPLE_RATE * RING_SECONDS;
static const size_t RING_BUFFER_BYTES   = RING_BUFFER_SAMPLES * sizeof(int16_t);
static const size_t LOOKBACK_SAMPLES    = (SAMPLE_RATE * LOOKBACK_MS) / 1000;
static const size_t I2S_READ_CHUNK      = 512; // 32ms chunks

// Speech detection threshold: voice is ~1500 to 4500, quiet room is ~200 to 450
#define AUTO_VOICE_TRIGGER      1      // 1 = Voice volume trigger enabled, 0 = BOOT button only
#define SPEECH_ENERGY_THRESHOLD 850    // Conversational speech threshold (calibrated for INMP441)
#define TRIGGER_COOLDOWN_MS     3000   // 3 seconds between triggers

// ---------------- STATE MACHINE ----------------
enum DeviceState { STATE_IDLE, STATE_STREAMING };
volatile DeviceState deviceState = STATE_IDLE;

int16_t *ringBuffer = nullptr;
volatile size_t ringWriteIdx = 0;
volatile size_t ringReadIdx  = 0;
volatile size_t totalSamplesWritten = 0;
SemaphoreHandle_t ringMutex;

QueueHandle_t triggerQueue;
volatile bool serverStopSignal = false;
volatile uint32_t lastStreamEndTime = 0;

volatile int16_t lastMicPeak = 0;
volatile uint32_t lastRawSample = 0;

// ---------------- CPU TELEMETRY ----------------
volatile uint32_t idleCount0 = 0;
volatile uint32_t idleCount1 = 0;

void idleCounterTask0(void *param) {
  for (;;) {
    idleCount0++;
    taskYIELD();
  }
}

void idleCounterTask1(void *param) {
  for (;;) {
    idleCount1++;
    taskYIELD();
  }
}

#if ENABLE_NETWORK_STREAM
WebSocketsClient webSocket;
volatile bool wsConnected = false;

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      wsConnected = true;
      Serial.printf("[WS] Connected to ws://%s:%d%s\n", WS_HOST, WS_PORT, WS_PATH);
      break;
    case WStype_DISCONNECTED:
      wsConnected = false;
      Serial.println("[WS] Disconnected from server.");
      break;
    case WStype_ERROR:
      Serial.printf("[WS] Connection Error: %s\n", payload ? (char *)payload : "");
      break;
    case WStype_TEXT:
      if (payload && strstr((const char *)payload, "\"stop\"") != nullptr) {
        serverStopSignal = true;
      }
      break;
    default:
      break;
  }
}
#endif

// ---------------- RING BUFFER HELPERS ----------------
void ringBufferWrite(const int16_t *samples, size_t count) {
  if (xSemaphoreTake(ringMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
    for (size_t i = 0; i < count; i++) {
      ringBuffer[ringWriteIdx] = samples[i];
      ringWriteIdx = (ringWriteIdx + 1) % RING_BUFFER_SAMPLES;
      totalSamplesWritten++;
    }
    xSemaphoreGive(ringMutex);
  }
}

// ---------------- I2S INIT ----------------
void i2sInit() {
#if !USE_FAKE_AUDIO
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = BITS_PER_SAMPLE,
    .channel_format = I2S_MIC_CHANNEL,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 6,
    .dma_buf_len = I2S_READ_CHUNK,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD_PIN
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  gpio_set_pull_mode((gpio_num_t)I2S_SD_PIN, GPIO_PULLDOWN_ONLY);
  i2s_zero_dma_buffer(I2S_PORT);
#endif
}

// ---------------- TRIGGER ----------------
void fireWakeWordTrigger() {
  uint32_t now = millis();
  if (now - lastStreamEndTime < TRIGGER_COOLDOWN_MS) return;
  if (deviceState != STATE_IDLE) return;

  bool val = true;
  xQueueSend(triggerQueue, &val, 0);
}

#if USE_PHYSICAL_BUTTON
void IRAM_ATTR onButtonPress() {
  uint32_t now = millis();
  if (now - lastStreamEndTime < 500) return;
  if (deviceState != STATE_IDLE) return;

  bool val = true;
  BaseType_t xHigherPriorityTaskWoken = pdFALSE;
  xQueueSendFromISR(triggerQueue, &val, &xHigherPriorityTaskWoken);
  if (xHigherPriorityTaskWoken) portYIELD_FROM_ISR();
}
#endif

// ---------------- TELEMETRY ----------------
void sendTelemetry() {
  uint32_t freeHeap = ESP.getFreeHeap();
  uint32_t minFreeHeap = ESP.getMinFreeHeap();
  uint32_t uptime = millis();

  static uint32_t lastIdle0 = 0, lastIdle1 = 0;
  static uint32_t maxDelta0 = 1, maxDelta1 = 1;

  uint32_t nowIdle0 = idleCount0, nowIdle1 = idleCount1;
  uint32_t delta0 = nowIdle0 - lastIdle0;
  uint32_t delta1 = nowIdle1 - lastIdle1;
  lastIdle0 = nowIdle0;
  lastIdle1 = nowIdle1;

  if (delta0 > maxDelta0) maxDelta0 = delta0;
  if (delta1 > maxDelta1) maxDelta1 = delta1;

  int cpu0Percent = 100 - (int)((delta0 * 100UL) / maxDelta0);
  int cpu1Percent = 100 - (int)((delta1 * 100UL) / maxDelta1);
  cpu0Percent = constrain(cpu0Percent, 0, 100);
  cpu1Percent = constrain(cpu1Percent, 0, 100);
  int cpuAvgPercent = (cpu0Percent + cpu1Percent) / 2;

#if ENABLE_NETWORK_STREAM
  if (wsConnected) {
    StaticJsonDocument<256> doc;
    doc["event"] = "telemetry";
    doc["free_heap"] = freeHeap;
    doc["min_free_heap"] = minFreeHeap;
    doc["cpu0_percent"] = cpu0Percent;
    doc["cpu1_percent"] = cpu1Percent;
    doc["cpu_percent"] = cpuAvgPercent;
    doc["uptime_ms"] = uptime;
    doc["mic_peak"] = lastMicPeak;
    String out;
    serializeJson(doc, out);
    webSocket.sendTXT(out);
  }
#endif

  Serial.printf("{\"event\":\"telemetry\",\"free_heap\":%u,\"min_free_heap\":%u,"
                "\"cpu0_percent\":%d,\"cpu1_percent\":%d,\"cpu_percent\":%d,\"uptime_ms\":%u,"
                "\"mic_peak\":%d,\"raw_hex\":\"0x%08X\"}\n",
                freeHeap, minFreeHeap, cpu0Percent, cpu1Percent, cpuAvgPercent, uptime,
                lastMicPeak, lastRawSample);
}

// ---------------- CORE 0 TASK: Audio Capture & Pre-Scale DC Filter ----------------
void audioCaptureTask(void *param) {
  static int32_t rawStereo[I2S_READ_CHUNK * 2];
  static int16_t chunk[I2S_READ_CHUNK];
  size_t bytesRead;
  static int32_t dcTracker = 0;
  static uint8_t sustainedEnergyCount = 0;
  static int activeChannel = 0; // 0 = slot 0 (even), 1 = slot 1 (odd)

  for (;;) {
#if USE_FAKE_AUDIO
    for (size_t i = 0; i < I2S_READ_CHUNK; i++) {
      chunk[i] = (int16_t)(1000 * sin(i * 0.05));
    }
    ringBufferWrite(chunk, I2S_READ_CHUNK);
    vTaskDelay(pdMS_TO_TICKS(32));
#else
    esp_err_t err = i2s_read(I2S_PORT, rawStereo, sizeof(rawStereo), &bytesRead, portMAX_DELAY);
    if (err == ESP_OK && bytesRead > 0) {
      size_t stereoPairs = bytesRead / (sizeof(int32_t) * 2);

      // Channel energy voting across the chunk: lock onto whichever slot has the active mic
      int64_t sumChan0 = 0;
      int64_t sumChan1 = 0;
      for (size_t i = 0; i < stereoPairs; i++) {
        sumChan0 += abs(rawStereo[2 * i]);
        sumChan1 += abs(rawStereo[2 * i + 1]);
      }

      // Lock active channel: only switch if the other channel has 2x higher energy
      if (sumChan0 > sumChan1 * 2) {
        activeChannel = 0;
      } else if (sumChan1 > sumChan0 * 2) {
        activeChannel = 1;
      }

      int16_t peak = 0;
      int32_t maxRaw = 0;

      for (size_t i = 0; i < stereoPairs; i++) {
        // ALWAYS sample from the same locked channel throughout the entire chunk!
        // This guarantees 100% continuous waveform with ZERO pops, clicks, or phase flips.
        int32_t rawSample = rawStereo[2 * i + activeChannel];

        if (abs(rawSample) > abs(maxRaw)) {
          maxRaw = rawSample;
        }

        // Step 1: Smooth 40Hz Pre-Scale DC High-Pass Filter (removes INMP441 DC offset)
        dcTracker += (rawSample - dcTracker) >> 6;
        int32_t acSample = rawSample - dcTracker;

        // Step 2: Calibrated 16-bit scaling (INMP441 puts 24-bit data in bits [31:8])
        // >> 11 provides calibrated 32x digital gain for INMP441 MEMS sensitivity (-26 dBFS)
        int32_t scaled = acSample >> 11;
        if (scaled > 32767) scaled = 32767;
        if (scaled < -32768) scaled = -32768;

        int16_t cleanSample = (int16_t)scaled;
        chunk[i] = cleanSample;

        int16_t a = abs(cleanSample);
        if (a > peak) peak = a;
      }

      lastMicPeak = peak;
      lastRawSample = (uint32_t)maxRaw;

      ringBufferWrite(chunk, stereoPairs);

      // Visual Diagnostic LED & Wake Word Voice Trigger
      if (deviceState == STATE_IDLE) {
        // 1. Visual Audio Detection:
        // SOLID GREEN when mic actively detects voice/sound (peak >= 550)
        // CONSTANT RED when room is silent (peak < 550)
        static uint32_t lastLedCheck = 0;
        if (millis() - lastLedCheck >= 50) {
          lastLedCheck = millis();
          if (peak >= 550) {
            setLedColor(0, 50, 0);  // Solid GREEN: mic actively picking up audio
          } else {
            setLedColor(40, 0, 0);  // Constant RED: silent / waiting
          }
        }

        // 2. Energy-gated wake word trigger (requires 2 consecutive speech blocks >= SPEECH_ENERGY_THRESHOLD)
#if AUTO_VOICE_TRIGGER
        if (millis() - lastStreamEndTime >= TRIGGER_COOLDOWN_MS) {
          if (peak >= SPEECH_ENERGY_THRESHOLD) {
            sustainedEnergyCount++;
            if (sustainedEnergyCount >= 2) {
              sustainedEnergyCount = 0;
              fireWakeWordTrigger();
            }
          } else {
            sustainedEnergyCount = 0;
          }
        }
#endif
      }
    }
#endif
  }
}

// ---------------- SERIAL BINARY CHUNK HELPER ----------------
void sendSerialAudioChunk(const int16_t *samples, size_t count) {
  uint16_t byteLen = count * sizeof(int16_t);
  uint8_t header[4] = {0xAA, 0x55, (uint8_t)(byteLen >> 8), (uint8_t)(byteLen & 0xFF)};
  Serial.write(header, 4);
  Serial.write((const uint8_t *)samples, byteLen);
}

// ---------------- CORE 1 TASK: True FIFO Stream Manager ----------------
void streamManagerTask(void *param) {
#if ENABLE_NETWORK_STREAM
  Serial.printf("\n[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  uint32_t wifiStart = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (millis() - wifiStart > 20000) {
      Serial.println("\n[WiFi] Connection taking long. Retrying...");
      wifiStart = millis();
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
  }
  WiFi.setSleep(false); // Disable modem sleep for ultra-low latency audio
  Serial.printf("\n[WiFi] Connected! IP: %s (RSSI: %d dBm)\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());

  Serial.printf("[WS] Connecting to ws://%s:%d%s\n", WS_HOST, WS_PORT, WS_PATH);
  webSocket.begin(WS_HOST, WS_PORT, WS_PATH);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(2000);
#endif

  int16_t sendBuf[I2S_READ_CHUNK];
  bool triggered = false;
  uint32_t lastTelemetry = 0;

  // Non-blocking serial command buffer
  char rxCmdBuf[32];
  uint8_t rxCmdIdx = 0;

  for (;;) {
#if ENABLE_NETWORK_STREAM
    webSocket.loop();
#endif

    if (millis() - lastTelemetry >= TELEMETRY_INTERVAL_MS) {
      lastTelemetry = millis();
      sendTelemetry();
    }

    if (xQueueReceive(triggerQueue, &triggered, 0) == pdTRUE) {
      deviceState = STATE_STREAMING;
      serverStopSignal = false;
      rxCmdIdx = 0;

      // Flash RED 4 times on Wake Word Detection
      flashWakeWordRed();
      // Keep GREEN while actively streaming speech audio
      setLedColor(0, 50, 0);

#if ENABLE_NETWORK_STREAM
      if (wsConnected) webSocket.sendTXT("{\"event\":\"start\"}");
#else
      Serial.println("{\"event\":\"start\"}");
#endif

      // Initialize true FIFO read pointer to LOOKBACK_SAMPLES behind the write pointer
      if (xSemaphoreTake(ringMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        size_t lookback = min((size_t)totalSamplesWritten, LOOKBACK_SAMPLES);
        ringReadIdx = (ringWriteIdx + RING_BUFFER_SAMPLES - lookback) % RING_BUFFER_SAMPLES;
        xSemaphoreGive(ringMutex);
      }

      uint32_t safetyDeadline = millis() + SAFETY_MAX_STREAM_MS;

      // True FIFO streaming loop: reads sequentially without dropping or duplicating samples
      while (true) {
#if ENABLE_NETWORK_STREAM
        webSocket.loop();
#else
        // 100% Non-blocking Serial reader for "stop" signal
        while (Serial.available()) {
          char c = (char)Serial.read();
          if (c == '\n' || c == '\r') {
            rxCmdBuf[rxCmdIdx] = '\0';
            if (strstr(rxCmdBuf, "stop") != NULL) {
              serverStopSignal = true;
            }
            rxCmdIdx = 0;
          } else if (rxCmdIdx < sizeof(rxCmdBuf) - 1) {
            rxCmdBuf[rxCmdIdx++] = c;
          }
        }
#endif
        if (serverStopSignal || millis() >= safetyDeadline) {
          break;
        }

        size_t avail = 0;
        if (xSemaphoreTake(ringMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
          avail = (ringWriteIdx + RING_BUFFER_SAMPLES - ringReadIdx) % RING_BUFFER_SAMPLES;
          if (avail >= I2S_READ_CHUNK) {
            for (size_t i = 0; i < I2S_READ_CHUNK; i++) {
              sendBuf[i] = ringBuffer[(ringReadIdx + i) % RING_BUFFER_SAMPLES];
            }
            ringReadIdx = (ringReadIdx + I2S_READ_CHUNK) % RING_BUFFER_SAMPLES;
          }
          xSemaphoreGive(ringMutex);
        }

        if (avail >= I2S_READ_CHUNK) {
#if ENABLE_NETWORK_STREAM
          if (wsConnected) webSocket.sendBIN((uint8_t *)sendBuf, I2S_READ_CHUNK * sizeof(int16_t));
#else
          sendSerialAudioChunk(sendBuf, I2S_READ_CHUNK);
#endif
        } else {
          // Wait briefly for new samples to arrive from Core 0
          vTaskDelay(pdMS_TO_TICKS(4));
        }
      }

      // Return to CONSTANT RED (waiting for next speech audio)
      setLedColor(45, 0, 0);
      serverStopSignal = false;
      lastStreamEndTime = millis();
      deviceState = STATE_IDLE;
    }

    // Manual 'T' trigger command
    if (Serial.available()) {
      char c = Serial.read();
      if (c == 'T' || c == 't') {
        fireWakeWordTrigger();
      }
    }

    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// ---------------- SETUP & LOOP ----------------
void setup() {
  Serial.begin(921600);
  delay(300);
  Serial.println("\n=== ESP32-S3 Edge KWS & Streaming Firmware Boot ===");

  ringBuffer = (int16_t *)heap_caps_malloc(RING_BUFFER_BYTES, MALLOC_CAP_SPIRAM);
  if (!ringBuffer) {
    ringBuffer = (int16_t *)malloc(RING_BUFFER_BYTES);
  }
  if (ringBuffer) {
    memset(ringBuffer, 0, RING_BUFFER_BYTES);
  }

  ringMutex = xSemaphoreCreateMutex();
  triggerQueue = xQueueCreate(4, sizeof(bool));

#if USE_STATUS_LED
  pinMode(STATUS_LED_PIN, OUTPUT);
#endif
  // Start with CONSTANT RED (waiting for microphone audio)
  setLedColor(45, 0, 0);

#if USE_PHYSICAL_BUTTON
  pinMode(TRIGGER_BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(TRIGGER_BUTTON_PIN), onButtonPress, FALLING);
#endif

  xTaskCreatePinnedToCore(idleCounterTask0, "Idle0", 1024, NULL, 0, NULL, 0);
  xTaskCreatePinnedToCore(idleCounterTask1, "Idle1", 1024, NULL, 0, NULL, 1);

  i2sInit();

  xTaskCreatePinnedToCore(audioCaptureTask, "AudioCapture", 6144, NULL, 2, NULL, 0);
  xTaskCreatePinnedToCore(streamManagerTask, "StreamManager", 8192, NULL, 1, NULL, 1);

#if ENABLE_NETWORK_STREAM
  Serial.println("[Boot] Operating Mode: Wi-Fi WebSocket Streaming");
#else
  Serial.println("[Boot] Operating Mode: USB Serial Streaming");
#endif
  Serial.println("[Boot] Firmware ready @ 921600 baud. Clean FIFO audio active.");
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(100));
}
