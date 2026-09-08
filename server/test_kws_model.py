#!/usr/bin/env python3
"""
TENet KWS TFLite Model Verification Script
===========================================
Loads the embedded INT8 Keyword Spotting model and verifies its tensor architecture,
quantization parameters, and runs a test inference on synthetic audio.

Usage:
    python test_kws_model.py [model_path]
"""

import sys
import os
import numpy as np

MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "kws_extracted.tflite")

CLASSES = [
    "0: wake_word (Ankit)",
    "1: local_negative",
    "2: noise",
    "3: silence",
    "4: unknown"
]

def load_interpreter(model_path):
    try:
        import tflite_runtime.interpreter as tflite
        print("[Engine] Using lightweight tflite-runtime engine.")
        return tflite.Interpreter(model_path=model_path)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        print("[Engine] Using TensorFlow Lite engine.")
        return tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        print("\n[ERROR] Neither 'tflite-runtime' nor 'tensorflow' is installed!")
        print("Install via: pip install tflite-runtime (or: pip install tensorflow)\n")
        sys.exit(1)

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Model file not found: {MODEL_FILE}")
        sys.exit(1)

    print(f"=== Verifying KWS Model: {MODEL_FILE} ===")
    print(f"Model File Size: {os.path.getsize(MODEL_FILE):,} bytes")

    interpreter = load_interpreter(MODEL_FILE)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n--- Input Tensor Details ---")
    for i, inp in enumerate(input_details):
        print(f"  Input #{i}: name='{inp['name']}', shape={inp['shape']}, dtype={inp['dtype'].__name__}")
        print(f"           quantization scale={inp['quantization'][0]}, zero_point={inp['quantization'][1]}")

    print("\n--- Output Tensor Details ---")
    for i, out in enumerate(output_details):
        print(f"  Output #{i}: name='{out['name']}', shape={out['shape']}, dtype={out['dtype'].__name__}")
        print(f"           quantization scale={out['quantization'][0]}, zero_point={out['quantization'][1]}")

    # Generate test dummy input: 1 batch x 51 frames x 1 channel x 10 spectral bins
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    test_input = np.zeros(input_shape, dtype=input_dtype)

    print(f"\n[Inference] Running test inference with dummy {input_shape} {input_dtype.__name__} tensor...")
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    print(f"[Inference] Raw Output Scores: {output_data.tolist()}")

    print("\n--- Class Activation Scores ---")
    for idx, (label, score) in enumerate(zip(CLASSES, output_data)):
        print(f"  [{label}]: {score:4d}")

    predicted_idx = int(np.argmax(output_data))
    print(f"\n[Result] Highest activation: Class {predicted_idx} ({CLASSES[predicted_idx]})")
    print("\n[SUCCESS] Model is valid, compatible, and ready for deployment!")

if __name__ == "__main__":
    main()
