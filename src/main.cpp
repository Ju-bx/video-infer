#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>

std::mutex cout_mutex;

using Clock = std::chrono::steady_clock;

const int INPUT_W = 640;
const int INPUT_H = 640;

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


std::vector<float> preprocess(const cv::Mat& bgr) {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

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

struct Detection {
    float cx, cy, w, h;
    float score;
    int class_id;
};

std::vector<Detection> decode(const float* out, int num_boxes,
                              int num_classes, float conf_thresh) {
    std::vector<Detection> dets;

    for (int i = 0; i < num_boxes; ++i) {

        float best_score = 0;
        int best_class = -1;
        for (int c = 0; c < num_classes; ++c) {
            float s = out[(4 + c) * num_boxes + i];
            if (s > best_score) {
                best_score = s;
                best_class = c;
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
    float uni = (ax2 - ax1)*(ay2 - ay1) + (bx2 - bx1)*(by2 - by1) - inter;                    // 空格 F

    if (uni <= 0) return 0.0f;
    return inter / uni;
}

std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh) {
    //按分數由高到低排序
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<Detection> keep;                      // 留下來的框
    std::vector<bool> removed(dets.size(), false);    // 記錄每個框有沒有被刪掉

    //一個一個看
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;

        keep.push_back(dets[i]);                      // 留下第 i 個

        // 把後面跟它重疊太多的刪掉
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (dets[j].class_id == dets[i].class_id &&
                iou(dets[i], dets[j]) >= iou_thresh) {//IoU 超過門檻
                removed[j] = true;//標記第 j 個為已刪除
            }
        }
    }
    return keep;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& dets) {
    // ① 換算比例：原圖是 640 的幾倍
    float sx = (float)img.cols / INPUT_W;     // 寬的比例
    float sy = (float)img.rows / INPUT_H;                        // 空格 A：高的比例

    for (const Detection& d : dets) {
        // ② 換成左上角、右下角，再乘上比例
        int x1 = (int)((d.cx - d.w / 2) * sx);
        int y1 = (int)((d.cy - d.h / 2) * sy);
        int x2 = (int)((d.cx + d.w / 2) * sx);
        int y2 = (int)((d.cy + d.h / 2) * sy);

        // ③ 畫框
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 255, 0), 2);

        // ④ 寫上類別和分數，例如 "bus 0.90"
        std::string label = COCO_NAMES[d.class_id] + " " +
                            std::to_string(d.score).substr(0, 4);
        cv::putText(img, label, cv::Point(x1, std::max(y1 - 5, 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
}

std::vector<Detection> detect(Ort::Session& session, const cv::Mat& frame, Timing& t){
    auto t0 = Clock::now();
    std::vector<float> input = preprocess(frame);
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
    auto t2 = Clock::now();

    std::vector<Detection> dets = decode(out, 8400, 80, 0.25f);
    std::vector<Detection>nms_result = nms(dets, 0.7f);
    auto t3 = Clock::now();

    t.pre_ms   = ms_between(t0, t1);
    t.infer_ms = ms_between(t1, t2);
    t.post_ms  = ms_between(t2, t3);

    return nms_result;
}

void process_stream(Ort::Session& session, const std::string& video_path, int stream_id) {
    // 打開影片
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "cannot open video: " << video_path << "\n";
        return;
    }

    // 準備輸出影片
    double fps = cap.get(cv::CAP_PROP_FPS);
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::string out_name = "result_" + std::to_string(stream_id) + ".avi";
    cv::VideoWriter writer(out_name,
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           fps, cv::Size(w, h));

    // 逐格處理 + 計時：跟原本完全一樣
    cv::Mat frame;
    int frame_count = 0;
    double total_pre = 0, total_infer = 0, total_post = 0, total_frame = 0;
    while (true) {
        auto f0 = Clock::now();
        cap >> frame;
        if (frame.empty()) break;

        Timing t;
        std::vector<Detection> dets = detect(session, frame, t);
        total_pre   += t.pre_ms;
        total_infer += t.infer_ms;
        total_post  += t.post_ms;

        draw_detections(frame, dets);
        writer.write(frame);

        auto f1 = Clock::now();
        total_frame += ms_between(f0, f1);
        frame_count++;
    }

    std::lock_guard<std::mutex> lock(cout_mutex);

    // 印統計：每一行前面加上是第幾路
    double avg_frame = total_frame / frame_count;
    std::cout << "[stream " << stream_id << "] frames: " << frame_count
              << "  infer: " << total_infer / frame_count << " ms"
              << "  per frame: " << avg_frame << " ms"
              << "  FPS: " << 1000.0 / avg_frame << "\n";
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.onnx> <video> [num_streams] [intra_threads]\n";
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
        }

        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto name  = session.GetOutputNameAllocated(i, allocator);
            auto shape = session.GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
            std::cout << "output : " << name.get() << " "
                      << shape_to_string(shape) << "\n";
        }

        // ① 從指令讀要開幾路，沒給就是 1 路
        int num_streams = (argc >= 4) ? std::stoi(argv[3]) : 1;

        auto start = Clock::now();

        // ② 開 N 個 thread
        std::vector<std::thread> threads;
        for (int i = 0; i < num_streams; ++i) {
            threads.emplace_back(process_stream, std::ref(session), std::string(argv[2]), i);
        }

        // ③ 等所有 thread 做完
        for (std::thread& t : threads) {
            t.join();
        }

        auto end = Clock::now();
        std::cout << "total time: " << ms_between(start, end) / 1000.0 << " s\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}