#include "draw.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

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