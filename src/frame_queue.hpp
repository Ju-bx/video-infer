#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include "common.hpp"

class FrameQueue {
public:
    FrameQueue(size_t max_size) {
        max_size_ = max_size;
    }

    void push(const Frame& f) {
        {
            std::lock_guard<std::mutex> lock(m_);
            if (q_.size() >= max_size_) {
                q_.pop();
                dropped_++;
            }
            q_.push(f);
        }
        cv_.notify_one();
    }

    bool pop(Frame& out) {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [this] { return !q_.empty() || closed_; });

        if (q_.empty()) return false;

        out = q_.front(); 
        q_.pop();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    int dropped() {
        std::lock_guard<std::mutex> lock(m_);
        return dropped_;
    }

private:
    std::queue<Frame> q_;
    std::mutex m_;
    std::condition_variable cv_;
    size_t max_size_;
    bool closed_ = false;
    int dropped_ = 0;
};