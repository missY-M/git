一、⭐ utils/inference_utils.py

1、get_dnn_model(arg: str)
输入：param arg: 模型名字
输出：return: 对应的名字
作用：获取DNN模型
2、model_partition(model, index)
输入：传入模型、模型划分点
输出：return edge_model, cloud_model 划分之后的边缘模型以及云端模型
作用：model_partition函数可以将一个整体的model,划分成两个部分
    划分的大致思路：
        如选定第index层对模型进行划分 ，则代表在第index后对模型进行划分
        将index层之前的层包括第index层 - 封装进edge_model中交给边缘端设备推理
        将index之后的层 - 封装进cloud_model交给云端设备推理

3、show_model_constructor(model,skip=True)
输入:传入模型、是否需要跳过 ReLU BatchNorm Dropout等层
输出：展示DNN各层结构
作用：展示DNN各层结构
检查当前层是不是 ReLU（激活层）、BatchNorm2d（批归一化层）或 Dropout（随机失活层）
如果是这三种层之一，直接执行 continue 跳过它，不对它进行编号和打印！
在深度学习模型（如 AlexNet 或 VGG）中，真正的“大件”往往是卷积层（Conv2d）和全连接层（Linear），它们占据了模型绝大部分的计算量和耗时。  而激活层（ReLU）、批归一化（BatchNorm）和 Dropout 只是紧跟在卷积层后面的“小挂件”。如果把这些小挂件也当成一个独立的切分点去算网络传输，在工程上不仅没有意义，还会极大地增加搜索切分点算法的复杂度。 


4、show_features(model, input_data, device, epoch_cpu=50, epoch_gpu=100, skip=True, save=False, sheet_name="model", path=None)
输入：DNN模型，输入数据，设备，cpu推理次数轮数、gpu推理次数、是否跳过、是否保存、表格名字、表格路径
输出：return: None
作用：可以输出DNN各层的性质,并将其保存在excel表格中,输出的主要性质如下：
    ["index", "layerName", "computation_time(ms)", "output_shape", "transport_num", "transport_size(MB)","accumulate_time(ms)"]
    [DNN层下标，层名字，层计算时延，层输出形状，需要传输的浮点数数量，传输大小，从第1层开始的累计推理时延]

调用warmUp函数：推理之前对设备进行预热；warmUp调用warmUpGpu(model, input_data, device, epoch)或warmUpCpu(model, input_data, device, epoch)函数
调用recordTime函数记录DNN模型或者DNN层的推理时间，根据设备分发到不同函数上进行计算，recordTime函数调用recordTimeGpu(model, input_data, device, epoch)或recordTimeCpu(model, input_data, device, epoch)函数。



二、 ⭐ predictor/predictor_utils.py
作用：
它通过线性回归模型对各种 DNN 层（如卷积、深度卷积、全连接、池化、归一化等）的推理时延进行预测。核心流程包括：解析输入层特征→调用对应的层类型预测函数（如 get_conv2d_lat）→构造特征向量→通过 predict_latency 调用 线性回归预测器 得到时延。


主要函数:
1、model_training_linear(filepath,threshold,get_datasets_func,model_path,save=False)
    param filepath:数据文件路径
    param threshold:阈值（判断“预测是否正确”的误差阈值）
    param get_datasets_func: 读取数据集的函数
    它负责把文件变成：data_features, data_targets
    | 输出            | 含义     |
    | data_features | 输入特征 X |
    | data_targets  | 预测目标 Y |

    param model_path: 模型参数保存位置
    param save: 是否保存模型参数
    return: model（一个训练好的线性回归模型）

2、kernel_predictor_creator(kernel_kind, device, predictor_dict):
根据不同 layer 类型（conv / fc / pool 等）+ 运行设备（edge/cloud），自动加载或训练一个“该类型layer的延迟预测模型”。
"""
    根据传入的kernel_kind对DNN层的推理时延进行预测
    input : [['HW','kernel','stride','Cin','Cout','FLOPs']]
    output : ['latency']
    :param kernel_kind : 0 - conv2d , 1 - dw_conv2d , 2 - linear, 3 - maxPool2d , 4 - avgPool2d, 5 - BatchNorm
    :param device : "edge" or "cloud"
    :param predictor_dict: 记录预测器是否已经被加载的全局字典（用来避免重复加载/训练模型）
    :return: 指定的预测器 predictor （为“某一类 layer + 某个设备”准备一个 latency 预测模型）
    """

3、predict_latency(features, kernel_kind, device, predictor_dict):

 """
    加载预测器并预测时延
    :param features: input features
    :param kernel_kind: 0 - conv2d , 1 - dw_conv2d , 2 - linear, 3 - maxPool2d , 4 - avgPool2d, 5 - BatchNorm
    :param device : cloud or edge
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency
    """
    根据输入的 layer 特征 + kernel 类型 + 设备，调用对应预测器，输出该层运行时间（latency）


4、predict_kernel_latency(input_features,layer, device, predictor_dict):
    param input_features: input x
    param layer: layer instance
    param device : cloud or edge
    param predictor_dict: 记录预测器是否已经被加载的全局字典
    return: latency
<eg>:
    get_conv2d_lat
        ↓
    features (HW², Cout)
        ↓
    predict_latency
        ↓
    kernel_predictor_creator
        ↓
    Linear Regression Model
“layer → 对应预测模型 → latency” 的路由分发

根据 layer 类型（Conv / Linear / Pool / BN 等），自动选择对应的 latency 预测函数，实现统一的 DNN layer-level 性能建模入口。


5、predict_model_latency(x, model,device, predictor_dict):
"""
    通过预测每一层的时延 来预测一个可循环结构的推理时延
    :param model: 传入的模型 (可能是list或者单个layer)
    :param x: 输入数据 (tensor)
    :param device: edge or cloud
    :param predictor_dict: 记录预测器是否已经被加载的全局字典
    :return: latency 整个模型的预测延迟（ms）
    """

    这个函数做的是：
    通过递归遍历模型的每一层，累加每一层的预测 latency，从而得到整个 DNN 的推理时间
👉 “逐层模拟 forward + 逐层累计 latency”
    predict_model_latency 通过递归遍历 DNN 的每一层，同时模拟 forward 过程和累加每层的预测延迟，从而估计整个模型的推理时间。


三、predictor/kernel_flops.py

"""
FLOPs参数可以帮助更好地调整预测器 - 这里没有用到
可以进一步用来提升模型预测性能
"""

计算不同DNN 层（Linear / Conv / Depthwise Conv）的 FLOPs（计算量），用于后续 latency 预测模型的特征输入
可以把它理解为：
❗“把神经网络层变成可量化计算成本的数字”

| 层类型     | FLOPs 复杂度                            |
| ------- | ------------------------------------ |
| Linear  | O(in × out)                          |
| Conv2D  | O(k² × Cin × Cout × HW²)             |
| DW Conv | O(k² × Cin × HW² + Cin × Cout × HW²) |


目标：理解模型切分和延迟预测。