#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat a(2, 2, CV_8UC1, cv::Scalar(0));   // 一張 2×2 的灰階小圖，全部是 0

    cv::Mat b = a;          // 方法 1：用 =
    cv::Mat c = a.clone();  // 方法 2：用 clone()

    a.at<uchar>(0, 0) = 99; // 把 a 左上角的像素改成 99

    std::cout << "a = " << (int)a.at<uchar>(0, 0) << "\n";
    std::cout << "b = " << (int)b.at<uchar>(0, 0) << "\n";
    std::cout << "c = " << (int)c.at<uchar>(0, 0) << "\n";
    return 0;
}