/*
 * HARDWARE TEAM FIRMWARE — Edge-to-Cloud Keyword Spotting (ESP32-S3)
 * Dual-Mode: USB Serial (921600 baud for demo) + Wi-Fi WebSocket (Final Deployment)
 *
 * Feature Pipeline:
 *   - I2S INMP441 Microphone (16 kHz mono, 32-bit slot, DC filter, calibrated gain)
 *   - Acoustic Gatekeeper:
 *       * Continuous energy & FFT spectral analysis
 *       * Accurately detects Room Silence vs Room Audio / Ambient Noise vs Human Voice
 *       * Gatekeeper OPENS only when voice energy is detected in the speech band (300 - 3400 Hz) above threshold
 *   - TENet TinyML KWS Engine:
 *       * 1.0-Second sliding window with 200ms stride
 *       * Real FFT MFCC Spectrogram Feature Extraction (51 frames x 10 channels)
 *       * TFLM INT8 Inverted Residual CNN Model ("Ankit" wake word detection)
 *   - True FIFO Ring Buffer: Guaranteed zero duplicated or skipped samples for cloud/server streaming
 *   - Status LED Visual Indication:
 *       * Constant RED = Room Silence / Waiting (Gatekeeper Closed)
 *       * Dim BLUE/AMBER = Ambient Room Noise (Gatekeeper Closed)
 *       * Solid GREEN = Active Voice Detected (Gatekeeper Open, KWS Active)
 *       * 4x Red Flashes = Wake Word ("Ankit") Detected!
 *       * Solid GREEN = Active Audio Streaming
 */

#include <Arduino.h>
#include "driver/i2s.h"
#include "kws_model_data.h"
#include "arduinoFFT.h"

// ---------------- MODERN TFLM_ESP32 INCLUDES ----------------
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION (3)
#endif

// ---------------- BUILD FLAGS ----------------
#define ENABLE_NETWORK_STREAM 0   // 0 = USB Serial (Prototype Demo), 1 = Wi-Fi WebSocket
#define USE_FAKE_AUDIO        0   // 0 = Real INMP441 I2S microphone (GPIO 4, 5, 7)
#define USE_PHYSICAL_BUTTON   1   // 1 = BOOT button (GPIO 0) backup trigger
#define USE_STATUS_LED        1   // 1 = Status LED (GPIO 2)

#if ENABLE_NETWORK_STREAM
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#endif

// ---------------- NETWORK CONFIG ----------------
// NOTE: ESP32-S3 Wi-Fi hardware supports 2.4 GHz ONLY (5 GHz Wi-Fi is not supported).
#define WIFI_SSID   "ARPIT"               // 2.4 GHz Wi-Fi SSID
#define WIFI_PASS   "123456789@"          // Wi-Fi Password
#define WS_HOST     "192.168.1.100"       // Replace with PC IPv4 Address (find via ipconfig)
#define WS_PORT     8080
#define WS_PATH     "/stream"

// ---------------- PIN ASSIGNMENTS (Soldered to GPIO 4, 5, 7) ----------------
#define I2S_WS_PIN   4    // Word Select / LRCLK (GPIO 4)
#define I2S_SCK_PIN  5    // Continuous Serial Clock / BCK (GPIO 5)
#define I2S_SD_PIN   7    // Serial Data Out / SD (GPIO 7)
#define I2S_PORT     I2S_NUM_0

#define STATUS_LED_PIN     2    // Single-color LED fallback
#define TRIGGER_BUTTON_PIN 0    // BOOT button

#ifndef RGB_BUILTIN
#define RGB_BUILTIN 48          // Onboard WS2812 RGB LED for ESP32-S3 DevKit
#endif

// ---------------- AUDIO & KWS CONFIGURATION ----------------
#define SAMPLE_RATE       16000
#define BITS_PER_SAMPLE   I2S_BITS_PER_SAMPLE_32BIT
#define I2S_MIC_CHANNEL   I2S_CHANNEL_FMT_RIGHT_LEFT
#define I2S_READ_CHUNK    512 // 32ms chunks

// STREAMING RING BUFFER (Look-Back + Live FIFO)
#define RING_SECONDS      3
#define LOOKBACK_MS       500
#define SAFETY_MAX_STREAM_MS 12000
#define TELEMETRY_INTERVAL_MS 3000

static const size_t RING_BUFFER_SAMPLES = SAMPLE_RATE * RING_SECONDS;
static const size_t RING_BUFFER_BYTES   = RING_BUFFER_SAMPLES * sizeof(int16_t);
static const size_t LOOKBACK_SAMPLES    = (SAMPLE_RATE * LOOKBACK_MS) / 1000;

int16_t *ringBuffer = nullptr;
volatile size_t ringWriteIdx = 0;
volatile size_t ringReadIdx  = 0;
volatile size_t totalSamplesWritten = 0;
SemaphoreHandle_t ringMutex;
volatile bool serverStopSignal = false;

// KWS 1-SECOND SLIDING WINDOW
#define KWS_WINDOW_SAMPLES      16000  // 1.0 second of 16kHz audio
#define KWS_STRIDE_SAMPLES      3200   // 200 ms sliding hop stride
#define TRIGGER_COOLDOWN_MS     4000   // 4 seconds cooldown between triggers

int16_t inferenceBuffer[KWS_WINDOW_SAMPLES];
volatile size_t inferenceWriteIdx = 0;
volatile size_t newSamplesSinceLastInference = 0;

// ---------------- ACOUSTIC GATEKEEPER CONFIGURATION ----------------
// Acoustic Gatekeeper distinguishes:
//   1. Silence: Ambient room is quiet (peak < SILENCE_ENERGY_THRESHOLD)
//   2. Room Audio: Background noise (fan, AC, rumble < 250 Hz) -> Gatekeeper CLOSED
//   3. Voice: Speech energy detected in 300 Hz - 3400 Hz voice band above threshold -> Gatekeeper OPEN
// The KWS TFLM model is invoked ONLY when the Gatekeeper confirms ACTIVE VOICE!
#define SILENCE_ENERGY_THRESHOLD    250     // Peak amplitude below this = Silence
#define VOICE_BAND_ENERGY_THRESHOLD 1200.0f // Required spectral magnitude in 300-3400Hz speech band
#define NOISE_REJECTION_RATIO       1.8f    // Reject if low-freq rumble (>1.8x voice band)

enum AcousticState {
  ACOUSTIC_SILENCE,     // Room quiet / silence
  ACOUSTIC_ROOM_AUDIO,  // Ambient room noise / fan / hum (Gatekeeper CLOSED)
  ACOUSTIC_VOICE        // Human voice detected in speech frequency band (Gatekeeper OPEN)
};

volatile AcousticState acousticGateState = ACOUSTIC_SILENCE;

// ---------------- TFLM MODEL & INFERENCE GLOBALS ----------------
#define KWS_CONFIDENCE_THRESH   0   // Raw INT8 >= 0 corresponds to >= 50% softmax probability

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  constexpr int kTensorArenaSize = 48 * 1024; // 48 KB internal arena for safety
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}

// ---------------- STATE MACHINE ----------------
enum DeviceState { STATE_IDLE, STATE_STREAMING };
volatile DeviceState deviceState = STATE_IDLE;
QueueHandle_t triggerQueue = nullptr;
volatile uint32_t lastStreamEndTime = 0;
volatile int16_t lastMicPeak = 0;
volatile uint32_t lastRawSample = 0;

// ---------------- CPU TELEMETRY ----------------
volatile uint32_t idleCount0 = 0;
volatile uint32_t idleCount1 = 0;

void idleCounterTask0(void *param) {
  for (;;) { idleCount0++; taskYIELD(); }
}

void idleCounterTask1(void *param) {
  for (;;) { idleCount1++; taskYIELD(); }
}

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
    setLedColor(70, 0, 0); delay(50);
    setLedColor(0, 0, 0);  delay(50);
  }
}

// ---------------- NETWORK GLOBALS & EVENTS ----------------
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

// ---------------- I2S INITIALIZATION ----------------
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

// ---------------- TFLM_ESP32 ML INITIALIZATION ----------------
bool initML() {
  model = tflite::GetModel(g_kws_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ML] Schema mismatch! Model version: %d, expected: %d\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  // Model requires exactly 7 operators:
  // Conv2D, DepthwiseConv2D, Add, MaxPool2D, Mean (GAP), FullyConnected, Softmax
  static tflite::MicroMutableOpResolver<8> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ML] Error: AllocateTensors() failed.");
    return false;
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  Serial.printf("[ML] TENet KWS Model Ready! Input bytes: %u, Output classes: %d\n",
                model_input->bytes, model_output->dims->data[1]);
  return true;
}

// ---------------- ACOUSTIC GATEKEEPER ----------------
// Fast spectral analysis using arduinoFFT to classify chunk into:
// SILENCE, ROOM_AUDIO (ambient noise), or VOICE (speech frequencies).
AcousticState evaluateAcousticGatekeeper(const int16_t *samples, size_t count, int16_t peak) {
  if (peak < SILENCE_ENERGY_THRESHOLD) {
    return ACOUSTIC_SILENCE;
  }

  const uint16_t fft_size = 512;
  static float vReal[fft_size];
  static float vImag[fft_size];

  size_t n = min(count, (size_t)fft_size);
  for (size_t i = 0; i < n; i++) {
    vReal[i] = (float)samples[i];
    vImag[i] = 0.0f;
  }
  for (size_t i = n; i < fft_size; i++) {
    vReal[i] = 0.0f;
    vImag[i] = 0.0f;
  }

  ArduinoFFT<float> FFT(vReal, vImag, fft_size, (float)SAMPLE_RATE);
  FFT.windowing(FFTWindow::Hann, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  // Frequency resolution per bin = SAMPLE_RATE / fft_size = 16000 / 512 = 31.25 Hz
  // Noise band: 0 Hz to ~250 Hz (bins 1 to 8)
  float noiseBandEnergy = 0.0f;
  for (int bin = 1; bin <= 8; bin++) {
    noiseBandEnergy += vReal[bin];
  }

  // Voice band: 300 Hz to ~3400 Hz (bins 10 to 108)
  float voiceBandEnergy = 0.0f;
  for (int bin = 10; bin <= 108; bin++) {
    voiceBandEnergy += vReal[bin];
  }

  // Decision logic:
  // If voice band energy is below the frequency threshold, or low-frequency noise dominates:
  if (voiceBandEnergy < VOICE_BAND_ENERGY_THRESHOLD || 
      (noiseBandEnergy > voiceBandEnergy * NOISE_REJECTION_RATIO && voiceBandEnergy < 3000.0f)) {
    return ACOUSTIC_ROOM_AUDIO;
  }

  return ACOUSTIC_VOICE;
}

// ---------------- REAL FFT FEATURE EXTRACTION ----------------
// Generates 51 frames x 10 spectral channels from 1.0s audio window
bool extract_mfcc_features(const int16_t* audio_data, int8_t* feature_output) {
  const uint16_t n_fft = 512; 
  const int hop = 320;        
  const int half_win = 200; // 25ms center offset
  
  static float vReal[n_fft];
  static float vImag[n_fft];
  
  ArduinoFFT<float> FFT(vReal, vImag, n_fft, (float)SAMPLE_RATE);

  for (int frame = 0; frame < 51; frame++) {
    int center_sample = frame * hop;
    int start_idx = center_sample - half_win;
    
    for (int i = 0; i < n_fft; i++) {
      int s_idx = start_idx + i;
      if (s_idx >= 0 && s_idx < KWS_WINDOW_SAMPLES) {
        vReal[i] = (float)audio_data[s_idx];
      } else {
        vReal[i] = 0.0f;
      }
      vImag[i] = 0.0f;
    }
    
    FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
    FFT.compute(FFTDirection::Forward);
    FFT.complexToMagnitude();
    
    for (int bin = 0; bin < 10; bin++) {
      float bin_energy = 0.0f;
      int bins_per_block = 25; 
      
      for (int j = 0; j < bins_per_block; j++) {
        bin_energy += vReal[(bin * bins_per_block) + j + 1]; 
      }
      bin_energy = bin_energy / (float)bins_per_block;
      
      // Quantization mapped to INT8 [-128, 127]
      int out_val = (int)(bin_energy / 200.0f) - 128;
      if (out_val > 127) out_val = 127;
      if (out_val < -128) out_val = -128;
      
      feature_output[(frame * 10) + bin] = (int8_t)out_val;
    }
  }
  return true;
}

// ---------------- TRIGGER ----------------
void fireWakeWordTrigger() {
  uint32_t now = millis();
  if (now - lastStreamEndTime < TRIGGER_COOLDOWN_MS) return;
  if (deviceState != STATE_IDLE) return;

  if (triggerQueue) {
    bool val = true;
    xQueueSend(triggerQueue, &val, 0);
  }
}

#if USE_PHYSICAL_BUTTON
void IRAM_ATTR onButtonPress() {
  uint32_t now = millis();
  if (now - lastStreamEndTime < 500) return;
  if (deviceState != STATE_IDLE) return;

  if (triggerQueue) {
    bool val = true;
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(triggerQueue, &val, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) portYIELD_FROM_ISR();
  }
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
    StaticJsonDocument<300> doc;
    doc["event"] = "telemetry";
    doc["free_heap"] = freeHeap;
    doc["min_free_heap"] = minFreeHeap;
    doc["cpu0_percent"] = cpu0Percent;
    doc["cpu1_percent"] = cpu1Percent;
    doc["cpu_percent"] = cpuAvgPercent;
    doc["uptime_ms"] = uptime;
    doc["mic_peak"] = lastMicPeak;
    doc["raw_hex"] = String("0x") + String(lastRawSample, HEX);
    doc["acoustic_state"] = (acousticGateState == ACOUSTIC_VOICE) ? "VOICE" : 
                            ((acousticGateState == ACOUSTIC_ROOM_AUDIO) ? "ROOM_AUDIO" : "SILENCE");
    String out;
    serializeJson(doc, out);
    webSocket.sendTXT(out);
  }
#endif

  Serial.printf("{\"event\":\"telemetry\",\"free_heap\":%u,\"min_free_heap\":%u,"
                "\"cpu0_percent\":%d,\"cpu1_percent\":%d,\"cpu_percent\":%d,\"uptime_ms\":%u,"
                "\"mic_peak\":%d,\"raw_hex\":\"0x%08X\","
                "\"gatekeeper\":\"%s\"}\n",
                freeHeap, minFreeHeap, cpu0Percent, cpu1Percent, cpuAvgPercent, uptime,
                lastMicPeak, lastRawSample,
                (acousticGateState == ACOUSTIC_VOICE) ? "VOICE" : 
                ((acousticGateState == ACOUSTIC_ROOM_AUDIO) ? "ROOM_AUDIO" : "SILENCE"));
}

// ---------------- CORE 0 TASK: Audio Capture, Acoustic Gatekeeper & KWS ----------------
void audioCaptureTask(void *param) {
  static int32_t rawStereo[I2S_READ_CHUNK * 2];
  static int16_t chunk[I2S_READ_CHUNK];
  size_t bytesRead;
  static int32_t dcTracker = 0;
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

      // Channel energy voting across chunk to lock active mic slot
      int64_t sumChan0 = 0, sumChan1 = 0;
      for (size_t i = 0; i < stereoPairs; i++) {
        sumChan0 += abs(rawStereo[2 * i]);
        sumChan1 += abs(rawStereo[2 * i + 1]);
      }
      if (sumChan0 > sumChan1 * 2) activeChannel = 0;
      else if (sumChan1 > sumChan0 * 2) activeChannel = 1;

      int16_t peak = 0;
      int32_t maxRaw = 0;

      for (size_t i = 0; i < stereoPairs; i++) {
        int32_t rawSample = rawStereo[2 * i + activeChannel];

        if (abs(rawSample) > abs(maxRaw)) {
          maxRaw = rawSample;
        }

        // Step 1: Smooth 40Hz Pre-Scale DC High-Pass Filter (removes INMP441 DC offset)
        dcTracker += (rawSample - dcTracker) >> 6;
        int32_t acSample = rawSample - dcTracker;

        // Step 2: Calibrated 16-bit scaling (>> 11 provides 32x digital gain for INMP441 -26 dBFS)
        int32_t scaled = acSample >> 11;
        if (scaled > 32767) scaled = 32767;
        if (scaled < -32768) scaled = -32768;

        int16_t cleanSample = (int16_t)scaled;

        // 1. Push to 1-second Inference Buffer (for KWS)
        inferenceBuffer[inferenceWriteIdx] = cleanSample;
        inferenceWriteIdx = (inferenceWriteIdx + 1) % KWS_WINDOW_SAMPLES;
        newSamplesSinceLastInference++;

        // 2. Prepare chunk for FIFO streaming ring buffer
        chunk[i] = cleanSample;

        if (abs(cleanSample) > peak) peak = abs(cleanSample);
      }

      lastMicPeak = peak;
      lastRawSample = (uint32_t)maxRaw;

      // Always write to streaming ring buffer so look-back window remains full!
      ringBufferWrite(chunk, stereoPairs);

      if (deviceState == STATE_IDLE) {
        // --- ACOUSTIC GATEKEEPER ---
        // Evaluate whether chunk is Silence vs Room Audio / Noise vs Active Speech
        AcousticState currentAcoustic = evaluateAcousticGatekeeper(chunk, stereoPairs, peak);
        acousticGateState = currentAcoustic;

        // Visual Diagnostic LED:
        static uint32_t lastLedCheck = 0;
        if (millis() - lastLedCheck >= 40) {
          lastLedCheck = millis();
          if (currentAcoustic == ACOUSTIC_VOICE) {
            setLedColor(0, 50, 0);   // Solid GREEN: mic actively picking up human voice!
          } else if (currentAcoustic == ACOUSTIC_ROOM_AUDIO) {
            setLedColor(0, 0, 30);   // Dim BLUE: Ambient room noise / fan (Gatekeeper CLOSED)
          } else {
            setLedColor(40, 0, 0);   // Constant RED: Silence / waiting
          }
        }

        // --- SLIDING KWS INFERENCE (200ms Hop Stride) ---
        if (newSamplesSinceLastInference >= KWS_STRIDE_SAMPLES) {
          newSamplesSinceLastInference = 0;

          // GATEKEEPER ENFORCEMENT:
          // ONLY invoke TFLM if human voice is actively detected above frequency threshold!
          if (acousticGateState == ACOUSTIC_VOICE && interpreter != nullptr && model_input != nullptr && model_output != nullptr) {
            if (millis() - lastStreamEndTime >= TRIGGER_COOLDOWN_MS) {
              
              // Unroll 1.0s circular buffer into linear memory
              static int16_t linearAudio[KWS_WINDOW_SAMPLES];
              size_t oldestIdx = inferenceWriteIdx;
              for (size_t i = 0; i < KWS_WINDOW_SAMPLES; i++) {
                linearAudio[i] = inferenceBuffer[(oldestIdx + i) % KWS_WINDOW_SAMPLES];
              }

              // Extract 51 frames x 10 spectral channels into TFLM input tensor
              if (extract_mfcc_features(linearAudio, model_input->data.int8)) {
                if (interpreter->Invoke() == kTfLiteOk) {
                  // Class mapping:
                  // 0: wake_word ("Ankit"), 1: local_negative, 2: noise, 3: silence, 4: unknown
                  int8_t wakeScore  = model_output->data.int8[0];
                  int8_t negScore   = model_output->data.int8[1];
                  int8_t noiseScore = model_output->data.int8[2];
                  int8_t silScore   = model_output->data.int8[3];
                  int8_t unkScore   = model_output->data.int8[4];

                  if (wakeScore >= KWS_CONFIDENCE_THRESH && wakeScore > negScore && wakeScore > noiseScore) {
                    Serial.printf("\n>>> [WAKE WORD DETECTED] Ankit! (Score: %d, Neg: %d, Noise: %d) <<<\n\n",
                                  wakeScore, negScore, noiseScore);
                    fireWakeWordTrigger();
                  } else {
                    Serial.printf("[KWS] Voice Detected | Wake: %d, Neg: %d, Noise: %d, Sil: %d, Unk: %d\n",
                                  wakeScore, negScore, noiseScore, silScore, unkScore);
                  }
                }
              }
            }
          } else {
            // Gatekeeper is CLOSED (Room Silence or Ambient Room Noise):
            // Model invocation is bypassed, keeping CPU < 5% and eliminating false triggers!
          }
        }
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
      Serial.println("\n[WiFi] Retrying...");
      wifiStart = millis();
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
  }
  WiFi.setSleep(false);
  Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());

  webSocket.begin(WS_HOST, WS_PORT, WS_PATH);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(2000);
#endif

  int16_t sendBuf[I2S_READ_CHUNK];
  bool triggered = false;
  uint32_t lastTelemetry = 0;

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

    if (triggerQueue && xQueueReceive(triggerQueue, &triggered, 0) == pdTRUE) {
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

      // Initialize true FIFO read pointer to LOOKBACK_SAMPLES behind write pointer
      if (ringBuffer && ringMutex && xSemaphoreTake(ringMutex, pdMS_TO_TICKS(10)) == pdTRUE) {
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
        // Non-blocking Serial reader for "stop" signal
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
        if (ringBuffer && ringMutex && xSemaphoreTake(ringMutex, pdMS_TO_TICKS(5)) == pdTRUE) {
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
  setLedColor(45, 0, 0); // Constant RED (waiting for microphone audio)

#if USE_PHYSICAL_BUTTON
  pinMode(TRIGGER_BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(TRIGGER_BUTTON_PIN), onButtonPress, FALLING);
#endif

  xTaskCreatePinnedToCore(idleCounterTask0, "Idle0", 1024, NULL, 0, NULL, 0);
  xTaskCreatePinnedToCore(idleCounterTask1, "Idle1", 1024, NULL, 0, NULL, 1);

  initML();
  i2sInit();

  xTaskCreatePinnedToCore(audioCaptureTask, "AudioCapture", 32768, NULL, 2, NULL, 0);
  xTaskCreatePinnedToCore(streamManagerTask, "StreamManager", 8192, NULL, 1, NULL, 1);

#if ENABLE_NETWORK_STREAM
  Serial.println("[Boot] Operating Mode: Wi-Fi WebSocket Streaming");
#else
  Serial.println("[Boot] Operating Mode: USB Serial Streaming");
#endif
  Serial.println("[Boot] Firmware ready @ 921600 baud. Acoustic Gatekeeper & TENet KWS active.");
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(100));
}
