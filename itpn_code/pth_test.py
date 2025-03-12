import cv2
import numpy as np
from mmcv.cnn import fuse_conv_bn
from mmdet.apis import inference_detector, init_detector, show_result_pyplot


# a = np.ones((800, 1333, 3), dtype=np.uint8)
# b = np.ones((700, 1333, 3), dtype=np.uint8)
# b = np.ones((1333, 800, 3), dtype=np.uint8)
# cv2.imwrite("800x1333.png", a)
# cv2.imwrite("700x1333.png", b)
# cv2.imwrite("1333x800.png", b)

config = "C:/gradenew/iTPN-main/det/configs/itpn/pixel_itpn_base_1x_ld090_dp005.py"
checkpoint = "C:/gradenew/iTPN-main/det/configs/epoch_2.pth"
img_path = "C:/gradenew/iTPN-main/det/demo/demo.jpg"
img_cropped_path = "C:/gradenew/itpn_det_rknn/itpn_det/demo_crop.jpg"
# d = cv2.imread(img_path)
# H, W, C = d.shape
# c = d[:W, :, :]
# cv2.imwrite("/root/workspaces/iTPN/det/demo/demo_crop.jpg", c)

model = init_detector(config, checkpoint, device="cpu")
model = fuse_conv_bn(model)

# test a single image
model_result = inference_detector(model, img_cropped_path)
model_result = model_result[0]
for i, anchor_set in enumerate(model_result):
    anchor_set = anchor_set[anchor_set[:, 4] >= 0.25]
    model_result[i] = anchor_set

# show the results
show_result_pyplot(
    model,
    img_cropped_path,
    model_result,
    score_thr=0.25,
    title='pytorch_result',
    out_file="result.png")

