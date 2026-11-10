#pragma once

#include <vector>
#include "common.hpp"

struct Track {
    int id;
    Detection box;
    int hits;
    int misses;
    float vx = 0;   // 每格在 x 方向移動多少
    float vy = 0;   // 每格在 y 方向移動多少
};

class Tracker {
public:
    std::vector<Track> update(const std::vector<Detection>& dets);   // ← 只有宣告

private:
    std::vector<Track> tracks_;
    int next_id_ = 0;
};