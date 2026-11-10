#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>

using Clock = std::chrono::steady_clock;

extern int INPUT_W;
extern int INPUT_H;

const int TARGET_CLASS = 0;
const int MAX_MISSES   = 3;
const int MIN_HITS     = 3;

inline double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct Timing {
    double pre_ms   = 0;   // 前處理
    double infer_ms = 0;   // 推論
    double post_ms  = 0;   // 後處理
};

struct Detection {
    float cx, cy, w, h;
    float score;
    int class_id;
};

struct Letterbox {
    float scale;   // 縮放比例
    int pad_x;
    int pad_y;
};

struct Frame {
    cv::Mat image;
    Clock::time_point capture_time;
    int index;
};