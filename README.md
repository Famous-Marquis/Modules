# Modules
基于pytorch2，保存经常用到的模块与代码

Module.py
------------
OpticalConv2d / OpticalConvTranspose2d: 为了模型训练能够更贴合实际光计算结果，加了噪声，以提升实际的鲁棒性。

utils.py
------------
RunManager：简单的用于记录模型训练日志、参数、权重的工具
OpticalCoSimulationEngine: 提取某层神经网络输入，供后续硬件计算。计算后，可重新注入，进行后续运算。
convert_to_optical_unet：替换特定层为光学层的函数（如果你直接用光学层来搭，就用不上这个了，主要针对module里的Unet。）
