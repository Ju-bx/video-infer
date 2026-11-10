#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <cmath>      // std::round
#include <deque>      // std::deque
#include <algorithm>  // std::sort、std::min、std::max

std::mutex cout_mutex;

using Clock = std::chrono::steady_clock;

int INPUT_W;
int INPUT_H;
const int TARGET_CLASS = 0;

const int MAX_MISSES = 3;
const int MIN_HITS   = 3; 

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

const std::vector<std::string> COCO_NAMES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

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

struct StreamStats {
    int captured = 0;
    int processed = 0;
    double total_infer = 0;
    double total_latency = 0;
    double total_pre = 0;
};

std::vector<float> preprocess(const cv::Mat& bgr, Letterbox& lb) {
    lb.scale = std::min((float)INPUT_W / bgr.cols, (float)INPUT_H/ bgr.rows);

    int new_w = (int)std::round(bgr.cols * lb.scale);
    int new_h = (int)std::round(bgr.rows * lb.scale);                                       // 空格 B

    lb.pad_x = (INPUT_W - new_w) / 2;
    lb.pad_y = (INPUT_H - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h));

    cv::Mat canvas(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(lb.pad_x, lb.pad_y, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);

    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    std::vector<float> chw(3 * INPUT_H * INPUT_W);
    const float* src = f32.ptr<float>(0);

    for (int c = 0; c < 3; ++c) {               
        for (int y = 0; y < INPUT_H; ++y) {     
            for (int x = 0; x < INPUT_W; ++x) { 
                int hwc_index = (y * INPUT_W + x) * 3 + c;  
                int chw_index = c * INPUT_H * INPUT_W + (y * INPUT_W + x); 
                chw[chw_index] = src[hwc_index];
            }
        }
    }

    return chw;
}

std::string shape_to_string(const std::vector<int64_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); i++){
        result += std::to_string(shape[i]);
        if (i+1 < shape.size()){
            result +=", ";
        }
    }
    result += "]";
    return result;
}

std::vector<Detection> decode(const float* out, int num_boxes,
                              int num_classes, float conf_thresh) {
    std::vector<Detection> dets;

    for (int i = 0; i < num_boxes; ++i) {
        float best_score = 0;
        int best_class = -1;
        if (TARGET_CLASS != -1){
            best_class = TARGET_CLASS;
            best_score = out[(4 + TARGET_CLASS) * num_boxes + i];
        }else{
            for (int c = 0; c < num_classes; ++c) {
                float s = out[(4 + c) * num_boxes + i];
                if (s > best_score) {
                    best_score = s;
                    best_class = c;
                }
            }
        }

        

        if (best_score < conf_thresh) continue;

        Detection d;
        d.cx = out[0 * num_boxes + i];
        d.cy = out[1 * num_boxes + i];
        d.w  = out[2 * num_boxes + i];
        d.h  = out[3 * num_boxes + i];
        d.score = best_score;
        d.class_id = best_class;
        dets.push_back(d);
    }
    return dets;
}

float iou(const Detection& a, const Detection& b) {

    float ax1 = a.cx - a.w / 2,  ay1 = a.cy - a.h / 2;
    float ax2 = a.cx + a.w / 2,  ay2 = a.cy + a.h / 2;
    float bx1 = b.cx - b.w / 2,  by1 = b.cy - b.h / 2;
    float bx2 = b.cx + b.w / 2,  by2 = b.cy + b.h / 2;

    //交集的左上角、右下角
    float ix1 = std::max(ax1, bx1);
    float iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2);
    float iy2 = std::min(ay2, by2);

    //交集面積（不重疊時是 0）
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;

    //  聯集面積 = A 的面積 + B 的面積 − 交集
    float uni = (ax2 - ax1)*(ay2 - ay1) + (bx2 - bx1)*(by2 - by1) - inter;

    if (uni <= 0) return 0.0f;
    return inter / uni;
}

std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh) {

    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<Detection> keep;
    std::vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;

        keep.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (dets[j].class_id == dets[i].class_id &&
                iou(dets[i], dets[j]) >= iou_thresh) {//IoU 超過門檻
                removed[j] = true;
            }
        }
    }
    return keep;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& dets) {

    for (const Detection& d : dets) {
        // 換成左上角、右下角，再乘上比例
        int x1 = (int)(d.cx - d.w / 2);
        int y1 = (int)(d.cy - d.h / 2);
        int x2 = (int)(d.cx + d.w / 2);
        int y2 = (int)(d.cy + d.h / 2);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);

        std::string label = COCO_NAMES[d.class_id] + " " +
                            std::to_string(d.score).substr(0, 4);
        cv::putText(img, label, cv::Point(x1, std::max(y1 - 5, 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

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

    std::vector<Track> update(const std::vector<Detection>& dets) {
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

private:
    std::vector<Track> tracks_;
    int next_id_ = 0;
};

void draw_tracks(cv::Mat& img, const std::vector<Track>& tracks) {

    for (const Track& t : tracks) {
        int x1 = (int)(t.box.cx - t.box.w / 2);
        int y1 = (int)(t.box.cy - t.box.h / 2);
        int x2 = (int)(t.box.cx + t.box.w / 2);
        int y2 = (int)(t.box.cy + t.box.h / 2);

        
        cv::Scalar color = (t.misses == 0)? cv::Scalar(0, 255, 0):cv::Scalar(0, 255, 255);

        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      color, 2);
        
    }
}

std::vector<Detection> detect(Ort::Session& session, const cv::Mat& frame, Timing& t){
    auto t0 = Clock::now();
    Letterbox lb;
    std::vector<float> input = preprocess(frame, lb);
    auto t1 = Clock::now();

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, INPUT_H, INPUT_W};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem,
        input.data(),
        input.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    const char* input_names[]  = {"images"};
    const char* output_names[] = {"output0"};

    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    float* out = outputs[0].GetTensorMutableData<float>();
    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto t2 = Clock::now();

    int num_boxes   = (int)out_shape[2];
    int num_classes = (int)out_shape[1] - 4;

    std::vector<Detection> dets = decode(out, num_boxes, num_classes, 0.25f);
    std::vector<Detection>nms_result = nms(dets, 0.7f);
    for (Detection& d : nms_result) {
        d.cx = (d.cx - lb.pad_x) / lb.scale;
        d.cy = (d.cy - lb.pad_y) / lb.scale;
        d.w  = d.w / lb.scale;
        d.h  = d.h / lb.scale;
    }
    auto t3 = Clock::now();

    t.pre_ms   = ms_between(t0, t1);
    t.infer_ms = ms_between(t1, t2);
    t.post_ms  = ms_between(t2, t3);

    return nms_result;
}

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

        // 讀一次影片的 FPS，給 writer 用
        cv::VideoCapture probe(argv[2]);
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
                                 std::ref(queues[i]), i, fps, std::ref(stats[i]));
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
                      << "  avg infer: "   << s.total_infer / s.processed   << " ms"
                      << "  avg latency: " << s.total_latency / s.processed << " ms"
                      << "  avg pre: " << s.total_pre / s.processed << " ms"
                      << "  FPS: "         << s.processed / total_s << "\n";
                      
        }
        std::cout << "total time: " << total_s << " s\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}