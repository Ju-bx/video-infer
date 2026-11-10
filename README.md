# video-infer

Real-time person detection in C++17 with ONNX Runtime and OpenCV, built step by step from a single-image prototype to multi-stream processing on a PC and real-time deployment on a Raspberry Pi 3B+.

The whole pipeline — letterbox preprocessing, ONNX Runtime inference, YOLO output decoding, IoU / NMS, a lightweight tracker, and drawing — is implemented by hand in C++ and was verified stage by stage against a Python reference.

**Results**

- **PC (4 vCPUs):** 4 simultaneous 10 FPS streams in real time, ~0.07% dropped frames, ~57 ms average capture-to-result latency.
- **Raspberry Pi 3B+ (4× Cortex-A53, 1 GB RAM):** one 10 FPS stream in real time (9.97 FPS processed, 0.25% dropped), up from 2.8 FPS in the first deployment, by profiling the bottleneck, choosing the input size, and detecting on keyframes.

![demo](docs/demo.gif)

## Features

- **Hand-written YOLO11n pipeline in C++**: letterbox resize, HWC→CHW, ONNX Runtime C++ API, decoding of the `[1, 84, N]` output, IoU and NMS
- **Model-agnostic input size**: input resolution and number of candidate boxes are read from the model, so 192–640 models run without code changes
- **Person-only detection** for the surveillance use case (reads the person score directly instead of scanning all 80 classes)
- **Real-time producer–consumer pipeline**: one capture thread per stream simulates a camera at the video's native FPS; a bounded, thread-safe `FrameQueue` drops the oldest frame when full to keep latency bounded
- **Multi-stream** processing with `std::thread`, sharing a single ONNX Runtime session
- **IoU tracker** for temporal smoothing: suppresses one-frame false positives and keeps tracks alive through short detector misses
- **Keyframe detection**: run the detector every N frames and reuse the last confirmed boxes in between
- **Optional video output**: drawing and encoding can be turned off, as on a real edge device that only reports detections
- Reports **latency, drop rate and throughput** per stream
- **Fault isolation**: exceptions are caught inside each worker thread, so one failing stream does not take down the others
- **Cross-platform build**: the same CMake project builds on x86-64 and on the Raspberry Pi (aarch64)

## Project structure

```
src/
├── common.hpp         shared types (Detection, Frame, Timing, Letterbox), clock helpers, constants
├── frame_queue.hpp    bounded thread-safe queue (mutex + condition_variable)
├── detector.hpp/.cpp  letterbox preprocessing, ONNX Runtime inference, decode, IoU, NMS
├── tracker.hpp/.cpp   IoU tracker
├── draw.hpp/.cpp      drawing detections and tracks
├── pipeline.hpp/.cpp  capture thread (producer) and inference thread (consumer)
└── main.cpp           argument parsing, session setup, thread launch, report
```

## Architecture

```
                 ┌──────────────────────────── per stream ────────────────────────────┐
 video / camera  │  capture thread          FrameQueue          inference thread
 ──────────────► │  read at native FPS ──►  [■]  ──►  every N frames:
                 │  timestamp + clone       bounded,    letterbox → ONNX Runtime → decode
                 │                          drop oldest → NMS → map to original coords → tracker
                 │                                      every frame (optional): draw → write
                 └─────────────────────────────────────────────────────────────────────┘
                                     shared Ort::Session (thread-safe Run)
```

- `FrameQueue` wraps `std::deque`, `std::mutex` and `std::condition_variable`; `pop()` blocks until a frame arrives or the producer calls `close()`.
- Latency is measured from the moment a frame is captured to the moment the inference thread finishes with it.
- The tracker matches tracks to detections by IoU; a track is shown after 3 hits and dropped after 5 consecutive misses. Only tracks matched in the latest detection are drawn.

## Build

Requirements: Ubuntu 24.04 or Raspberry Pi OS / Debian (64-bit), CMake ≥ 3.16, a C++17 compiler.

```bash
# OpenCV
sudo apt install -y libopencv-dev

# ONNX Runtime (prebuilt C++ library)
# x86-64:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
# Raspberry Pi (aarch64), extracted to ~/video-infer-deps/:
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-aarch64-1.20.1.tgz

# Export models on a PC (Python, one time; input size must be a multiple of 32)
python3 -m venv .venv && source .venv/bin/activate
pip install ultralytics onnx
mkdir -p models
yolo export model=yolo11n.pt format=onnx imgsz=416 && mv yolo11n.onnx models/yolo11n_416.onnx

# Build (defaults to Release, i.e. -O3)
cmake -S . -B build
cmake --build build          # on the Raspberry Pi: cmake --build build -j1 (1 GB RAM)
```

`CMakeLists.txt` picks the ONNX Runtime folder from the CPU architecture, so the same project builds on both machines. The build type defaults to `Release`; without optimization the hand-written preprocessing is about 2.3× slower (4.5 ms → 2.0 ms per frame at 416×416).

## Usage

```bash
./build/video_infer <model.onnx> <video> [num_streams] [intra_threads] [queue_size] [detect_every] [write_video]

# PC: 4 streams, 1 ONNX Runtime thread each, queue size 1
./build/video_infer models/yolo11n_416.onnx videos/vtest.avi 4 1 1

# Raspberry Pi: 1 stream, 3 threads, detect every 2nd frame, no video output
./build/video_infer models/yolo11n_224.onnx videos/vtest.avi 1 3 1 2 0
```

| Argument | Default | Meaning |
|---|---|---|
| `num_streams` | 1 | number of simulated camera streams |
| `intra_threads` | ONNX Runtime default | threads used inside one inference |
| `queue_size` | 3 | max frames buffered per stream |
| `detect_every` | 1 | run the detector every N frames, reuse the last boxes in between |
| `write_video` | 1 | `0` skips drawing and writing `result_<i>.avi` |

## Benchmarks: PC

**Environment:** AMD Ryzen 5 5600H (6C/12T) · VirtualBox VM with 4 vCPUs and 8 GB RAM · Ubuntu 24.04 · YOLO11n FP32 · OpenCV sample video `vtest.avi` (768×576, 10 FPS, 795 frames).

Numbers inside a VM depend heavily on host load. The final result below is the median of 3 runs with other host programs closed; the earlier tables were measured in noisier sessions and with an unoptimized build (which affects preprocessing and postprocessing, not ONNX Runtime inference), so compare trends within a table rather than absolute values across tables.

### Final configuration (v1.0)

416×416 input · letterbox · person only · tracker · 4 streams × 1 intra-op thread · queue size 1 · camera simulated at 10 FPS

| Metric | Value (median of 3 runs) |
|---|---|
| Inference / frame | 44.5 ms |
| Capture-to-result latency | 56.9 ms |
| Processed FPS / stream | ≈ 9.99 (input: 10) |
| Dropped frames | ≈ 0.07% (2–3 of 3180) |

### Input resolution (4 streams, real-time 10 FPS input)

| Input size | Inference / frame | Drop rate | Avg latency |
|---|---|---|---|
| 320 | 28 ms | 0.2% | 40 ms |
| **416** | **64 ms** | **1%** | **90 ms** |
| 480 | 104 ms | 24% | 162 ms |
| 640 | 212 ms | 60% | 293 ms |

Each frame has a 100 ms budget at 10 FPS. Once processing exceeds it (480 and above), the drop rate jumps sharply rather than growing gradually. 416 is the largest input that stays within budget, keeping more detail than 320, where small people flicker in and out. Going from 640 to 320 sped inference up 7.5× — more than the 4× reduction in pixels — because the CPU was no longer saturated and the smaller activations fit better in cache.

### Streams × intra-op threads (640 input, offline, as fast as possible)

| Streams × threads per inference | Total FPS | Inference / frame |
|---|---|---|
| 1 × 4 (default) | 11.4 | 59 ms |
| 4 × 4 (default) | 14.1 | 198 ms |
| 1 × 1 | 8.4 | 96 ms |
| **4 × 1** | **17.1** | 195 ms |

With ONNX Runtime's default thread count, 4 streams oversubscribe the 4 vCPUs (≈16 busy threads). Limiting each inference to 1 thread raises total throughput by 21%, but per-stream inference still slows from 96 to 195 ms because of shared resources (memory bandwidth, L3 cache, SMT siblings).

### Queue size (640 input, real-time 10 FPS input, 1 thread per inference)

| Streams | Queue size | Drop rate | Avg latency |
|---|---|---|---|
| 1 | 3 | 27% | 384 ms |
| 1 | 1 | 25% | 171 ms |
| 4 | 3 | 63% | 514 ms |
| 4 | 1 | 63% | 311 ms |

When inference cannot keep up, the drop rate is set by processing speed, not queue size. A shorter queue only reduces how long the processed frame waited (latency ≈ queue length × processing time).

### Keyframe detection (416, 4 streams, queue size 1)

| Detect every | Inference (detected frames) | Avg latency (all frames) | Drop rate |
|---|---|---|---|
| 1 | 73 ms | 95 ms | ≈ 1.5% |
| 2 | ≈ 61 ms | ≈ 42 ms | ≈ 0.03% |
| 3 | ≈ 62 ms | ≈ 31 ms | ≈ 0.06% |

Average latency drops below the inference time because reused frames finish in a few milliseconds. This average hides the fact that boxes on reused frames are up to (N−1) frames old, i.e. up to 100 ms at N = 2 and 200 ms at N = 3; both numbers are needed to describe the system.

## Benchmarks: Raspberry Pi 3B+

**Environment:** Raspberry Pi 3B+ (4× Cortex-A53 @ 1.4 GHz, 1 GB RAM) · Raspberry Pi OS 64-bit (Debian 13) · ONNX Runtime 1.20.1 aarch64 · YOLO11n FP32 · same video, 1 stream, queue size 1 · SoC temperature below 60 °C (no thermal throttling).

### Final configuration (v2.0)

224×224 input · 3 intra-op threads · detect every 2nd frame · no video output

| Metric | Value |
|---|---|
| Inference (detected frames) | 146 ms |
| Preprocessing | 13 ms |
| Processed FPS | 9.97 (input: 10) |
| Dropped frames | 0.25% (2 of 795) |
| Avg latency (all frames) | 107 ms |
| Box age on reused frames | up to 100 ms |

### Step 1: the first deployment (1 frame = 1 detection, with video output)

| Input | Threads | Inference | Processed FPS | Drop rate | Avg latency |
|---|---|---|---|---|---|
| 320 | 1 | 576 ms | 1.5 | 84% | — |
| 320 | 2 | 338 ms | 2.4 | 75% | — |
| 320 | 4 | 279 ms | 2.8 | 71% | 403 ms |
| 256 | 4 | 177 ms | 4.2 | 58% | 288 ms |
| 224 | 4 | 146 ms | 4.9 | 51% | 259 ms |
| 192 | 4 | 109 ms | 6.1 | 39% | 212 ms |

Inference is 10× slower than on the PC. Even at 192, a single detection does not fit in the 100 ms frame budget, and smaller inputs lose small, distant people.

### Step 2: keyframe detection (with video output)

| Input | Detect every | Inference | Processed FPS | Drop rate |
|---|---|---|---|---|
| 256 | 2 | 176 ms | 6.6 | 33% |
| 224 | 2 | 143 ms | 7.9 | 21% |
| 256 | 3 | 174 ms | 7.4 | 25% |

Detecting every 2nd frame gives each detection a 200 ms budget. 224 needs only ~157 ms, so it should not drop frames, yet it still dropped 21%. Something other than inference was taking the remaining time.

### Step 3: finding the real bottleneck (224, detect every 2, 3 threads)

| Change | Drop rate | Processed FPS | Conclusion |
|---|---|---|---|
| baseline (with video output) | 21% | 7.8 | |
| 3 instead of 4 ONNX Runtime threads | 21% | 7.8 | not CPU contention with the capture thread; at 224, the 4th thread gives no speed-up (143 vs 144 ms) |
| desktop (GUI) turned off | 23% | 7.7 | not memory; the idle desktop uses RAM, not CPU |
| **video output turned off** | **0.25%** | **9.97** | **encoding 768×576 MJPG every frame was the bottleneck** |

On the PC, writing the output video is cheap and never showed up. On the Pi it costs enough per frame to break the budget, and it runs on every frame, including the reused ones. An edge device normally sends detection results rather than re-encoded video, so video output became optional (`write_video = 0`). Moving it to a separate thread was rejected: the Pi's CPU is already fully used, so the work would only move to another core.

### Step 4: final comparison (no video output, 3 threads)

| Input | Detect every | Drop rate | Processed FPS | Avg latency |
|---|---|---|---|---|
| 224 | 1 | 38% | 6.2 | 212 ms |
| **224** | **2** | **0.25%** | **9.97** | **107 ms** |
| 256 | 2 | 4.9% | 9.5 | 139 ms |

224 at every frame drops 38%, matching the estimate 1 − 100 / 161 ms ≈ 38%. 256 at N = 2 needs ~195 ms of a 200 ms budget and drops 5% from small variations, so 224 at N = 2 is the largest setting that runs in real time.

### Lessons

- **Measure before optimizing.** INT8 quantization was the planned next step, but it would only have sped up inference; the actual bottleneck was video encoding.
- A cost that is negligible on a PC can dominate on an edge device, so profiling has to be repeated on the target hardware.
- Real-time processing has a hard budget (100 ms per frame at 10 FPS); the drop rate rises sharply once the budget is exceeded.
- An average can hide what matters: with keyframes, average latency improves, but boxes on reused frames are older.

## Known limitations

- **Crowded scenes**: when two people overlap heavily, NMS can treat their boxes as duplicates and keep only one.
- **Occlusion**: partially hidden people fall below the confidence threshold. The tracker bridges short gaps but not long ones.
- **No identity tracking**: the tracker uses position only, so a person passing behind a pole often gets a new track (ID switch). Track IDs are therefore not displayed; stable identities would need appearance features (e.g. DeepSORT) or a stronger association method (e.g. ByteTrack with a Kalman filter).
- **Keyframe detection** keeps boxes fixed between detections, so on fast motion they lag by up to (N−1) frames. Predicting positions with the tracker's velocity was tried and looked worse, so it is not used.
- **Small inputs** (224 on the Pi) miss small, distant people more often than 416.
- When a stream's inference thread fails, its capture thread keeps decoding until the video ends; a stop signal from consumer to producer would avoid the wasted work.

## Roadmap

- [x] v0.1 — single-image detection in C++, verified against Python
- [x] v0.2 — video inference with per-stage timing
- [x] v0.3 — multi-stream threads and real-time producer–consumer pipeline
- [x] v1.0 — resolution study, letterbox, person-only filter, IoU tracker
- [x] v1.1 — split into modules, exception handling in worker threads, Release build by default
- [x] v2.0 — deployment on Raspberry Pi 3B+, keyframe detection, bottleneck analysis, optional video output
- [ ] v2.1 — INT8 quantization, to run larger inputs (e.g. 320) on the Pi in real time
- [ ] rectangular model input (e.g. 320×416 for 4:3 video), ByteTrack comparison

## Model

YOLO11n pretrained on COCO, from [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), exported to ONNX.