#include "detector.hpp"
#include <algorithm>
#include <cmath> 

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