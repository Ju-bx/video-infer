#pragma once

#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "common.hpp"

std::vector<float> preprocess(const cv::Mat& bgr, Letterbox& lb);

std::string shape_to_string(const std::vector<int64_t>& shape);

std::vector<Detection> decode(const float* out, int num_boxes, int num_classes, float conf_thresh);

float iou(const Detection& a, const Detection& b);

std::vector<Detection> nms(std::vector<Detection> dets, float iou_thresh);

std::vector<Detection> detect(Ort::Session& session, const cv::Mat& frame, Timing& t);