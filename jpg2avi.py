import cv2
import os

def create_video_from_images(input_dir, output_video, frame_rate):
    # 获取图片列表并排序
    images = [img for img in os.listdir(input_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()  # 确保按照文件名顺序合并

    if not images:
        print("No images found in the directory.")
        return

    # 读取第一张图片以获取宽度和高度
    first_image_path = os.path.join(input_dir, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # 定义视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # 写入每一帧
    for image_name in images:
        image_path = os.path.join(input_dir, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to read {image_path}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_video}")

    # 播放生成的视频
    cap = cv2.VideoCapture(output_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Result Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # 按 'q' 退出播放
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_dir = "./result"  # 图片输入目录
    output_video = "output_result.avi"  # 输出视频路径
    frame_rate = 30  # 视频帧率

    create_video_from_images(input_dir, output_video, frame_rate)
