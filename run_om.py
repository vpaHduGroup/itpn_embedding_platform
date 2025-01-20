import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession

coco_classes = {
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

def bbox_overlap(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    overlap = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (
                bb_gt[3] - bb_gt[1]) - wh)
    return overlap


def cpu_nms(detections, iou_threshold):
    # sort descend
    detections = np.array(detections)  # [n, 6]
    idxes = np.argsort(-(detections[:, 4]))
    detections = list(detections[idxes])

    # nms
    tmp_detections = list()
    keep_detections = list()
    while len(detections) != 0:
        if len(detections) == 1:
            keep_detections.append(detections[0])
            break

        keep_detections.append(detections[0])
        tmp_detections.clear()
        for idx in range(1, len(detections)):
            iou = bbox_overlap(keep_detections[-1][0:4], detections[idx][0:4])
            if iou < iou_threshold:
                tmp_detections.append(detections[idx])
        detections = tmp_detections.copy()
    return keep_detections


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
    vis = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if use_norm:
        im = (im - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
        im = im.astype(np.float32)
    im = np.ascontiguousarray(np.transpose(im, axes=(2, 0, 1)))  # HWC -> CHW
    im = np.expand_dims(im, axis=0).copy()
    return im, vis


def main():
    om_file = "model.om"
    img_path = "./demo_crop.jpg"

    # Load model
    model = InferSession(0,om_file)

    # Read and preprocess image
    cv_image = cv2.imread(img_path)
    preprocessed_img, vis = preprocess(cv_image, (480, 480), use_norm=True)  # (1, 3, 480, 480)

    # Run inference
    infer_start = time.time()
    outputs = model.infer([preprocessed_img])
    bbox_preds, cls_scores, max_idxes = outputs[0], outputs[1], outputs[2]
    total_time = time.time() - infer_start

    print("cls_scores:", cls_scores.shape)
    print("bbox_preds:", bbox_preds.shape)
    print("max_idxes:", max_idxes.shape)

    num_proposals = cls_scores.shape[0]
    num_classes = cls_scores.shape[1] - 1

    detections = list()
    for idx in range(num_proposals):
        max_idx = max_idxes[idx][0]
        score = cls_scores[idx][max_idx]
        if score < 0.5:
            continue
        x1 = bbox_preds[idx][max_idx][0]
        y1 = bbox_preds[idx][max_idx][1]
        x2 = bbox_preds[idx][max_idx][2]
        y2 = bbox_preds[idx][max_idx][3]
        # clip
        x1 = x1 if x1 >= 0 else 0
        y1 = y1 if y1 >= 0 else 0
        x2 = x2 if x2 < cv_image.shape[1] else cv_image.shape[1] - 1
        y2 = y2 if y2 < cv_image.shape[0] else cv_image.shape[0] - 1
        label = int(max_idx)
        detections.append([x1, y1, x2, y2, score, label])

    keep_detections = cpu_nms(detections, 0.45)
    print(len(keep_detections))

    for dect in keep_detections:
        x1, y1, x2, y2, score, max_idx = dect
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        label = int(max_idx)
        class_name = coco_classes[label]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{class_name}:{score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    print(total_time)
    cv2.imwrite("result.jpg", vis)


if __name__ == "__main__":
    main()