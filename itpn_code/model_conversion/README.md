onnx->om需要使用特定的ATC命令

例如：atc --model=itpn.onnx --framework=5 --output=itpn --input_shape="images:1,3,224,224"  --soc_version=Ascend310B1  --insert_op_conf=aipp.cfg
