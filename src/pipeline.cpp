#include "pipeline.hpp"
#include "detector.hpp"
#include "tracker.hpp"
#include "draw.hpp"
#include <iostream>
#include <mutex>
#include <thread> 
#include <stdexcept> 




std::mutex cout_mutex;

void capture_loop(const std::string& video_path, FrameQueue& queue, int& captured) {
    int idx = 0;

    try {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("cannot open video: " + video_path);   // ② 改成丟出錯誤
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 10;
        auto interval = std::chrono::microseconds((long long)(1000000 / fps));

        auto next = Clock::now();
        cv::Mat img;
        while (cap.read(img)) {
            Frame f;
            f.image = img.clone();
            f.capture_time = Clock::now();
            f.index = idx++;
            queue.push(f);

            next += interval;
            std::this_thread::sleep_until(next);
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "capture error: " << e.what() << "\n";
    }

    captured = idx;
    queue.close();
}

void infer_loop(Ort::Session& session, FrameQueue& queue, int stream_id,
                double fps, StreamStats& stats) {
    std::string out_name = "result_" + std::to_string(stream_id) + ".avi";
    cv::VideoWriter writer;                         //建立一個空的 writer
    
    Tracker tracker;
    try {
        Frame f;
        while (queue.pop(f)) {

            Timing t;
            std::vector<Detection> dets = detect(session, f.image, t);
            std::vector<Track> tracks = tracker.update(dets);
            draw_tracks(f.image, tracks);

            if (!writer.isOpened()) {
                writer.open(out_name, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                            fps, f.image.size());
            }
            writer.write(f.image);

            stats.processed++;
            stats.total_infer   += t.infer_ms;
            stats.total_latency += ms_between(f.capture_time, Clock::now());
            stats.total_pre += t.pre_ms;
        }
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "[stream " << stream_id << "] inference error: " << e.what() << "\n";
    }
}