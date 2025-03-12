import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
import time
from rknn.api import RKNN

# Model from https://github.com/airockchip/rknn_model_zoo
ONNX_MODEL = 'final_sim.onnx'
RKNN_MODEL = f'final_sim-{int(time.time())}.rknn'
DATASET = './dataset_50.txt'


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True, verbose_file=f'{int(time.time())}.log')

    # pre-process config
    print('--> Config model')
    rknn.config(
        mean_values=[123.675, 116.28, 103.53], 
        std_values=[58.82, 58.82, 58.82], 
        quant_img_RGB2BGR=False,
        target_platform='rk3588',
        single_core_mode=False,
        enable_flash_attention=False,
        compress_weight=True,
        model_pruning=True,
        optimization_level=3,
        disable_rules=['fuse_focus_slice', 'gather_to_conv']
        )
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
