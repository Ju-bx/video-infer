#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <functional>          // std::ref
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "frame_queue.hpp"
#include "detector.hpp"
#include "pipeline.hpp"

int INPUT_W = 0;
int INPUT_H = 0;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.onnx> <video> [num_streams] [intra_threads] [queue_size]\n";
        return 1;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "video_infer");

        int intra_threads = (argc >= 5) ? std::stoi(argv[4]) : 0;

        Ort::SessionOptions options;
        if (intra_threads > 0) {
            options.SetIntraOpNumThreads(intra_threads);
        }
        Ort::Session session(env, argv[1], options);

        Ort::AllocatorWithDefaultOptions allocator;

        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto name  = session.GetInputNameAllocated(i, allocator);
            auto shape = session.GetInputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
            std::cout << "input : " << name.get() << " "
                      << shape_to_string(shape) << "\n";
            INPUT_H = (int)shape[2];
            INPUT_W = (int)shape[3];
        }

        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto name  = session.GetOutputNameAllocated(i, allocator);
            auto shape = session.GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
            std::cout << "output : " << name.get() << " "
                      << shape_to_string(shape) << "\n";

        }

        int num_streams   = (argc >= 4) ? std::stoi(argv[3])  : 1;
        size_t queue_size = (argc >= 6) ? std::stoul(argv[5]) : 3;
        int detect_every  = (argc >= 7) ? std::stoi(argv[6])  : 1;
        bool write_video = (argc > 7) ? std::stoi(argv[7]) != 0 : true;

        // 讀一次影片的 FPS，給 writer 用
        cv::VideoCapture probe(argv[2]);
        if (!probe.isOpened()) {
            std::cerr << "cannot open video: " << argv[2] << "\n";
            return 1;
        }
        double fps = probe.get(cv::CAP_PROP_FPS);
        probe.release();
        std::cout << "video FPS: " << fps << "  queue size: " << queue_size << "\n";

        // 每一路一個 FrameQueue、一份統計
        std::deque<FrameQueue> queues;
        for (int i = 0; i < num_streams; ++i) {
            queues.emplace_back(queue_size);
        }
        std::vector<StreamStats> stats(num_streams);

        // 每一路開兩個 thread擷取 + 推論
        auto start = Clock::now();
        std::vector<std::thread> threads;
        for (int i = 0; i < num_streams; ++i) {
            threads.emplace_back(capture_loop, std::string(argv[2]),
                                 std::ref(queues[i]), std::ref(stats[i].captured));
            threads.emplace_back(infer_loop, std::ref(session),
                     std::ref(queues[i]), i, fps, detect_every, std::ref(stats[i]), write_video);
        }

        //全部做完
        for (std::thread& t : threads) {
            t.join();
        }
        auto end = Clock::now();
        double total_s = ms_between(start, end) / 1000.0;

        //印出每一路的結果
        for (int i = 0; i < num_streams; ++i) {
            const StreamStats& s = stats[i];
            if (s.processed == 0) continue;
            std::cout << "[stream " << i << "]"
                      << "  captured: "    << s.captured
                      << "  processed: "   << s.processed
                      << "  dropped: "     << queues[i].dropped()
                      << "  avg infer: "   << s.total_infer / s.detected   << " ms"
                      << "  avg latency: " << s.total_latency / s.processed << " ms"
                      << "  avg pre: " << s.total_pre / s.detected << " ms"
                      << "  FPS: "         << s.processed / total_s << "\n";
                      
        }
        std::cout << "total time: " << total_s << " s\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}