#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include <vector>

// TODO 1：把形狀轉成字串，例如 {1, 3, 640, 640} → "[1, 3, 640, 640]"
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
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <model.onnx>\n";
        return 1;
    }

    try {
        // ① 建立 ONNX Runtime 的環境（整個程式只需要一個）
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "video_infer");

        // ② 載入模型，建立推論 session
        Ort::SessionOptions options;
        Ort::Session session(env, argv[1], options);

        // ③ 取名字時需要的記憶體配置器
        Ort::AllocatorWithDefaultOptions allocator;

        // ④ 印出所有輸入
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto name  = session.GetInputNameAllocated(i, allocator);
            auto shape = session.GetInputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
            std::cout << "input : " << name.get() << " "
                      << shape_to_string(shape) << "\n";
        }

        // TODO 2：用一樣的方式印出所有輸出
        //   提示：把 Input 換成 Output
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            auto name  = session.GetOutputNameAllocated(i, allocator);
            auto shape = session.GetOutputTypeInfo(i)
                                .GetTensorTypeAndShapeInfo()
                                .GetShape();
            std::cout << "output : " << name.get() << " "
                      << shape_to_string(shape) << "\n";
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}