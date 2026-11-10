#include "tracker.hpp"
#include "detector.hpp"

std::vector<Track> Tracker::update(const std::vector<Detection>& dets) {
    std::vector<bool> used(dets.size(), false);
    for (Track& t : tracks_){
        float prev_cx = t.box.cx;
        float prev_cy = t.box.cy;
        t.box.cx += t.vx;
        t.box.cy += t.vy;

        float best_iou = 0;
        int best_j = -1;
        for (size_t j = 0; j < dets.size();j++){
            
            if (used[j]) continue;
            float v = iou(dets[j],t.box);
            if (v > best_iou){
                best_iou = v;
                best_j = (int)j;
            }
        }
        if (best_j != -1 && best_iou > 0.2f){
            t.vx = dets[best_j].cx - prev_cx;
            t.vy = dets[best_j].cy - prev_cy;
            t.box = dets[best_j];
            t.hits++;
            t.misses = 0;
            used[best_j] = true;
        } else {
            t.misses++;
        }
    }
    for (size_t i = 0; i < used.size(); i++){
        if (!used[i]){
            Track nt; 
            nt.id = next_id_;
            next_id_++;
            nt.box = dets[i];
            nt.hits = 1;
            nt.misses = 0;
            tracks_.push_back(nt);
        }
    }

    std::vector<Track> alive;
    for (const Track& t : tracks_) {
        if (t.misses <= MAX_MISSES) {
            alive.push_back(t);
        }
    }
    tracks_ = alive;
    

    std::vector<Track> confirmed;
    for (const Track& t : tracks_) {
        if (t.hits >= MIN_HITS) {
            confirmed.push_back(t);
        }
    }
    return confirmed;
}