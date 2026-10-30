#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

const int INPUT_W = 640;
const int INPUT_H = 640;

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

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}