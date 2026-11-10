#pragma once 

#include <vector>
#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "tracker.hpp"

void draw_detections(cv::Mat& img, const std::vector<Detection>& dets);

void draw_tracks(cv::Mat& img, const std::vector<Track>& tracks);