#pragma once 

#include <string>
#include <onnxruntime_cxx_api.h>
#include "common.hpp"
#include "frame_queue.hpp"

struct StreamStats {
    int captured = 0;
    int processed = 0;
    double total_infer = 0;
    double total_latency = 0;
    double total_pre = 0;
    int detected = 0;
};

void capture_loop(const std::string& video_path, FrameQueue& queue, int& captured);

void infer_loop(Ort::Session& session, FrameQueue& queue, int stream_id,
                double fps, int detect_every, StreamStats& stats);