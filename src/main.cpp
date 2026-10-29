#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

const int INPUT_W = 640;
const int INPUT_H = 640;

std::vector<float> preprocess(const cv::Mat& bgr) {
    // ① 縮放成 640×640
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(INPUT_W, INPUT_H));

    // ② BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // ③ uint8（0~255）→ float（0~1）
    cv::Mat f32;
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    // ④ HWC → CHW
    std::vector<float> chw(3 * INPUT_H * INPUT_W);
    const float* src = f32.ptr<float>(0);   // 指向 HWC 資料的開頭

    // TODO：用三層迴圈（c、y、x），把 src 的資料搬到 chw
    //   chw[CHW 的位置] = src[HWC 的位置];
    for (int c = 0; c < 3; ++c) {               // 每個顏色
        for (int y = 0; y < INPUT_H; ++y) {     // 每一列
            for (int x = 0; x < INPUT_W; ++x) { // 每一行
                int hwc_index = (y * INPUT_W + x) * 3 + c;   // 在 src 的第幾格
                int chw_index = c * INPUT_H * INPUT_W + (y * INPUT_W + x);                      // 在 chw 的第幾格
                chw[chw_index] = src[hwc_index];             // 搬過去
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <model.onnx>\n";
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

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}