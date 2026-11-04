# video-infer

Real-time, multi-stream object detection in C++17 with ONNX Runtime and OpenCV, built step by step toward edge deployment on a Raspberry Pi.

The whole inference pipeline — preprocessing, ONNX Runtime inference, YOLO output decoding, IoU / NMS, and drawing — is implemented by hand in C++ and verified step by step against a Python reference.

> **Status:** work in progress (v0.3). See [Roadmap](#roadmap).

<!-- TODO: add a demo GIF of result_0.avi here -->

## Features

- **Hand-written YOLO11n pipeline in C++**: HWC→CHW preprocessing, ONNX Runtime C++ API, decoding of the `[1, 84, 8400]` output, IoU and NMS, box drawing
- **Verified against Python** at every stage (preprocessed tensor, raw model output, decoded boxes)
- **Video inference** with per-stage timing (preprocess / inference / postprocess)
- **Multi-stream processing** with `std::thread`, sharing a single ONNX Runtime session
- **Real-time producer–consumer pipeline**: a capture thread per stream simulates a camera at the video's native FPS; a bounded, thread-safe `FrameQueue` drops the oldest frame when full to keep latency bounded
- Reports **latency, drop rate and throughput** per stream

## Architecture

```
                 ┌──────────────────── per stream ────────────────────┐
 video / camera  │  capture thread          FrameQueue       inference thread
 ──────────────► │  read at native FPS ──►  [■ ■ ■]  ──►  preprocess → ONNX Runtime
                 │  timestamp + clone      bounded,       → decode → NMS → draw → write
                 │                         drop oldest
                 └─────────────────────────────────────────────────────┘
                                   shared Ort::Session (thread-safe Run)
```

- `FrameQueue` wraps `std::queue`, `std::mutex` and `std::condition_variable`; `pop()` blocks until a frame arrives or the producer calls `close()`.
- Latency is measured from the moment a frame is captured to the moment its detections are written.

## Build

Requirements: Ubuntu 24.04, CMake ≥ 3.16, a C++17 compiler.

```bash
# OpenCV
sudo apt install -y libopencv-dev

# ONNX Runtime (prebuilt C++ library)
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar xzf onnxruntime-linux-x64-1.20.1.tgz

# Export the model (Python, one time)
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics onnx
yolo export model=yolo11n.pt format=onnx imgsz=640
mkdir -p models && mv yolo11n.onnx models/

# Build
cmake -S . -B build
cmake --build build
```

## Usage

```bash
./build/video_infer <model.onnx> <video> [num_streams] [intra_threads] [queue_size]

# 4 streams, 1 ONNX Runtime thread per inference, queue size 1
./build/video_infer models/yolo11n.onnx videos/vtest.avi 4 1 1
```

| Argument | Default | Meaning |
|---|---|---|
| `num_streams` | 1 | number of simulated camera streams |
| `intra_threads` | ONNX Runtime default | threads used inside one inference |
| `queue_size` | 3 | max frames buffered per stream |

## Benchmarks

**Environment:** AMD Ryzen 5 5600H (6C/12T) · VirtualBox VM with 4 vCPUs and 8 GB RAM · Ubuntu 24.04 · YOLO11n FP32, 640×640 · OpenCV sample video `vtest.avi` (768×576, 10 FPS, 795 frames).
Numbers come from single runs inside a VM and vary by roughly ±10–15% between runs.

### Per-frame breakdown (single stream, offline)

| Stage | Time | Share |
|---|---|---|
| Inference | 61.2 ms | 67% |
| Preprocess | 16.8 ms | 18% |
| Read + draw + write | 11.1 ms | 12% |
| Postprocess (decode + NMS) | 2.1 ms | 2% |
| **Total** | **91.2 ms** | **≈ 11 FPS** |

### Streams × intra-op threads (offline, as fast as possible)

| Streams × threads per inference | Total FPS | Inference / frame |
|---|---|---|
| 1 × 4 (default) | 11.4 | 59 ms |
| 4 × 4 (default) | 14.1 | 198 ms |
| 1 × 1 | 8.4 | 96 ms |
| **4 × 1** | **17.1** | 195 ms |

Running 4 streams with ONNX Runtime's default thread count oversubscribes the 4 vCPUs (≈16 busy threads). Limiting each inference to 1 thread raises total throughput by 21%, but per-stream inference still slows from 96 ms to 195 ms because of shared resources (memory bandwidth, L3 cache, SMT siblings).

### Real-time input (camera simulated at 10 FPS, 1 thread per inference)

| Streams | Queue size | Processed FPS / stream | Drop rate | Avg latency |
|---|---|---|---|---|
| 1 | 3 | 7.3 | 27% | 384 ms |
| 1 | 1 | 7.5 | 25% | 171 ms |
| 4 | 3 | 3.6 | 63% | 514 ms |
| 4 | 1 | 3.7 | 63% | 311 ms |

When inference cannot keep up with the camera, the drop rate is set by processing speed, not by queue size. A shorter queue only reduces how long the processed frame waited — latency ≈ queue length × processing time — so queue size 1 cuts latency by more than half at the same drop rate.

## Roadmap

- [x] v0.1 — single-image detection in C++, verified against Python
- [x] v0.2 — video inference with per-stage timing
- [x] v0.3 — multi-stream threads and real-time producer–consumer pipeline
- [ ] v1.0 — optimizations (input resolution, faster preprocessing, letterbox, person-only filter, IoU tracker with frame skipping), repeated benchmarks, demo GIF
- [ ] v2 — deployment on Raspberry Pi 3B+ (ARM, 1 GB RAM)

## Model

YOLO11n pretrained on COCO, from [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), exported to ONNX.