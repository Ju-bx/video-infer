import glob
import sys
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (quantize_static, CalibrationDataReader,
                                      QuantFormat)

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 224
INPUT_W = SIZE
INPUT_H = SIZE

def preprocess(path):
    bgr = cv2.imread(path)



    scale = min(INPUT_W / bgr.shape[1], INPUT_H / bgr.shape[0])
    new_w = round(bgr.shape[1] * scale)
    new_h = round(bgr.shape[0] * scale)
    pad_x = (INPUT_W - new_w) // 2
    pad_y = (INPUT_H - new_h) // 2

    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((INPUT_H, INPUT_W, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    f32 = rgb.astype(np.float32) / 255.0

    chw = f32.transpose(2, 0, 1)
    batch_tensor = np.expand_dims(chw, axis=0)

    return batch_tensor

class Reader(CalibrationDataReader):
    def __init__(self, files, input_name):
        self.it = iter([{input_name: preprocess(f)} for f in files])
    def get_next(self):
        return next(self.it, None)

prep = f"models/yolo11n_{SIZE}_prep.onnx"
m = onnx.load(prep)
exclude = [n.name for n in m.graph.node
           if n.name.startswith("/model.23/")
           and "/cv2." not in n.name and "/cv3." not in n.name]
print(f"exclude {len(exclude)} nodes")

files = sorted(glob.glob("calib/coco128/images/train2017/*.jpg"))
quantize_static(prep,
                f"models/yolo11n_{SIZE}_int8.onnx",
                Reader(files, "images"),
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                nodes_to_exclude=exclude)
print(f"done: models/yolo11n_{SIZE}_int8.onnx")