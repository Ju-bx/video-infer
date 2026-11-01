#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

const int INPUT_W = 640;
const int INPUT_H = 640;

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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.onnx> <image>\n";
        return 1;
    }

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "video_infer");

        Ort::SessionOptions options;
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

        cv::Mat img = cv::imread(argv[2]);
        if (img.empty()) {
            std::cerr << "cannot read image: " << argv[2] << "\n";
            return 1;
        }
        std::cout << "image: " << img.cols << "x" << img.rows << "\n";

        std::vector<float> input = preprocess(img);
        std::cout << "input size: " << input.size() << "\n";

        const int plane = INPUT_H * INPUT_W;
        for (int c = 0; c < 3; ++c) {
            double sum = 0;
            for (int i = 0; i < plane; ++i) sum += input[c * plane + i];
            std::cout << "channel " << c << ": "
                      << input[c * plane + 0] << " "
                      << input[c * plane + 1] << " "
                      << input[c * plane + 2]
                      << "  mean=" << sum / plane << "\n";
        }


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

        std::cout << "output shape: " << shape_to_string(out_shape) << "\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "out[" << i << "] = " << out[i] << "\n";
        }

        std::vector<Detection> dets = decode(out, 8400, 80, 0.25f);
        std::cout << "detections: " << dets.size() << "\n";
        for (size_t k = 0; k < dets.size() && k < 10; ++k) {
            const Detection& d = dets[k];
            std::cout << "class=" << d.class_id << " score=" << d.score
                    << " cx=" << d.cx << " cy=" << d.cy
                    << " w=" << d.w << " h=" << d.h << "\n";
        }

        std::vector<Detection> final_dets = nms(dets, 0.7f);
        std::cout << "after NMS: " << final_dets.size() << "\n";
        for (const Detection& d : final_dets) {
            std::cout << "class=" << d.class_id << " score=" << d.score
                    << " cx=" << d.cx << " cy=" << d.cy
                    << " w=" << d.w << " h=" << d.h << "\n";
        }

        draw_detections(img, final_dets);
        cv::imwrite("result.jpg", img);
        std::cout << "saved result.jpg\n";

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}