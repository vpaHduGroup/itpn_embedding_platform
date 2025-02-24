## 样例介绍

以itpn网络模型为例，在rk3588以及华为Ascend310B1上进行部署，实现对图片进行物体检测和分类，并给出标定框和类别置信度，以及mAP的计算。
  
样例输入：图片。    
样例输出：图片物体检测，并且在图片上给出物体标注框，类别，置信度以及mAP值

## 获取模型
    
 可以通过以下网盘链接快速获取pt、onnx、rknn以及om模型：
 Network model stored in https://pan.quark.cn/s/56b3c5cf311a password：zTJc


## 第三方依赖安装
设置环境变量

absl-py                  2.1.0
addict                   2.4.0
anykeystore              0.2
apex                     0.1
blobfile                 3.0.0
cachetools               5.5.0
certifi                  2024.8.30
charset-normalizer       3.4.0
click                    8.1.7
coloredlogs              15.0.1
contourpy                1.1.1
cryptacular              1.6.2
cycler                   0.12.1
Cython                   3.0.11
deepspeed                0.6.5
defusedxml               0.7.1
docker-pycreds           0.4.0
einops                   0.8.0
exceptiongroup           1.2.2
filelock                 3.16.1
flatbuffers              24.3.25
fonttools                4.55.0
fsspec                   2024.10.0
ftfy                     6.2.3
gitdb                    4.0.11
GitPython                3.1.43
google-auth              2.36.0
google-auth-oauthlib     1.0.0
greenlet                 3.1.1
grpcio                   1.67.1
hjson                    3.1.0
humanfriendly            10.0
hupper                   1.12.1
idna                     3.10
importlib_metadata       8.5.0
importlib_resources      6.4.5
iniconfig                2.0.0
Jinja2                   3.1.4
joblib                   1.4.2
kiwisolver               1.4.7
lxml                     5.3.0
Markdown                 3.7
markdown-it-py           3.0.0
MarkupSafe               2.1.5
matplotlib               3.7.5
mdurl                    0.1.2
mkl-fft                  1.3.8
mkl-random               1.2.4
mkl-service              2.4.0
mmcv-full                1.6.0
mmdet                    2.24.1       
mmengine                 0.10.5
mpmath                   1.3.0
mypy                     1.13.0
mypy-extensions          1.0.0
networkx                 3.1
ninja                    1.11.1.2
numpy                    1.23.2
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1

特别注意mmdet、mmcv的版本下载,itpn网络模型源码中有要求,具体可看源码README.md

## 样例运行
--rknn3588平台
    1.以onnx2rknn.py为例实现模型转换，可能需要根据具体报错特别修改rknn.config中disable_rules的值
    2.通过run_onnx_map_max.py和run_rknn_map_max.py获得结果
--Ascend310B1平台
    1.以atc命令实现onnx2om模型转化，具体可看官方文档，例子如下：
    atc --model=final_sim.onnx --framework=5 --output=final_sim.om --soc_version=Ascend310B1
    2.run_om_map_max.py可根据run_rknn_map_max.py修改模型文件获得结果
