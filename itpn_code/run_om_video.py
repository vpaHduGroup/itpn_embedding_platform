# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
from ais_bench.infer.interface import InferSession

# COCO 类别标签
coco_classes ={
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "TV",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
} 
def preprocess(cv_image, target_size, use_norm=False):
    h, w, c = cv_image.shape
    top, bottom, left, right = 0, 0, 0, 0
    tw, th = target_size
    sw, sh = float(w) / tw, float(h) / th
    if sw > sh:
        s = sw
        nw = tw
        nh = int(h / s)
        top = 0
        bottom = th - nh - top
    else:
        s = sh
        nh = th
        nw = int(w / s)
        left = 0
        right = tw - nw - left

    im = cv2.resize(cv_image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if use_norm:
        im = (im - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        im = im.astype(np.float32)
    im = np.ascontiguousarray(np.transpose(im, axes=(2, 0, 1)))  # HWC -> CHW
    im = np.expand_dims(im, axis=0).copy()
    return im, (left, top, right, bottom)

def bbox_overlap(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    overlap = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return overlap

def cpu_nms(detections, iou_threshold):
    detections = np.array(detections)  # [n, 6]
    idxes = np.argsort(-detections[:, 4])
    detections = detections[idxes]
    tmp_detections = []
    keep_detections = []
    while len(detections) > 0:
        if len(detections) == 1:
            keep_detections.append(detections[0])
            break
        keep_detections.append(detections[0])
        tmp_detections.clear()
        for det in detections[1:]:
            iou = bbox_overlap(keep_detections[-1][:4], det[:4])
            if iou < iou_threshold:
                tmp_detections.append(det)
        detections = tmp_detections.copy()
    return keep_detections

def process_video(input_video, output_dir):
    # 加载模型
    om_file = "model.om"
    model = InferSession(0, om_file)

    # 创建视频捕获和写入
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_dir, fourcc, frame_rate, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理帧
        preprocessed_img, (left, top, right, bottom) = preprocess(frame, (480, 480), use_norm=True)

        # 推理
        outputs = model.infer([preprocessed_img])
        bbox_preds, cls_scores, max_idxes = outputs[0], outputs[1], outputs[2]

        # 计算缩放比例
        original_height, original_width = frame.shape[:2]
        resized_height = 480 - top - bottom
        resized_width = 480 - left - right
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        # 处理检测结果
        detections = []
        for idx in range(cls_scores.shape[0]):
            max_idx = max_idxes[idx][0]
            score = cls_scores[idx][max_idx]
            if score < 0.5:
                continue
            # 调整坐标：减去填充并缩放到原始尺寸
            x1 = (bbox_preds[idx][max_idx][0] - left) * scale_x
            y1 = (bbox_preds[idx][max_idx][1] - top) * scale_y
            x2 = (bbox_preds[idx][max_idx][2] - left) * scale_x
            y2 = (bbox_preds[idx][max_idx][3] - top) * scale_y
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(original_width - 1, int(round(x2)))
            y2 = min(original_height - 1, int(round(y2)))
            detections.append([x1, y1, x2, y2, score, max_idx])

        # 非极大值抑制
        keep_detections = cpu_nms(detections, 0.45)

        # 绘制检测结果
        for det in keep_detections:
            x1, y1, x2, y2, score, max_idx = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = coco_classes.get(int(max_idx), "Unknown")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 实时显示处理后的帧
        cv2.imshow("Processed Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 按 'q' 退出
            break
            
        # 写入帧
        video_writer.write(frame)

    cap.release()
    video_writer.release()

if __name__ == "__main__":
    input_video_path = "/home/HwHiAiUser/Downloads/itpn_om/videos/3s.mp4"
    output_video_dir = os.path.dirname(input_video_path)  # 获取输入视频所在的目录
    output_video_path = os.path.join(output_video_dir, "output_result.avi")  # 输出文件路径

    process_video(input_video_path, output_video_path)