import cv2
import os

# 创建用于保存帧的目录
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture('./videos/test_1.mp4')  # 读取视频

# 定义OpenCV的七种追踪算法
OPENCV_OBJECT_TRACKERS = {
    'boosting': cv2.legacy.TrackerBoosting_create,
    'csrt': cv2.legacy.TrackerCSRT_create,
    'kcf': cv2.legacy.TrackerKCF_create,
    'mil': cv2.legacy.TrackerMIL_create,
    'tld': cv2.legacy.TrackerTLD_create,
    'medianflow': cv2.legacy.TrackerMedianFlow_create,
    'mosse': cv2.legacy.TrackerMOSSE_create
}

trackers = cv2.legacy.MultiTracker_create()  # 创建追踪器
frame_count = 0  # 记录帧数

while True:
    flag, frame = cap.read()
    if frame is None:
        break

    frame_count += 1  # 更新帧计数

    # 将当前帧保存为图片
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图

    # 更新追踪器并绘制追踪框
    success, boxes = trackers.update(frame)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # 框选ROI区域
        roi = cv2.selectROI('frame', frame, showCrosshair=True, fromCenter=False)
        print(f"Selected ROI: {roi}")

        # 创建并添加新的追踪器
        tracker = OPENCV_OBJECT_TRACKERS['boosting']()
        trackers.add(tracker, frame, roi)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
