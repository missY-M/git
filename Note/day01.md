一、deployment.py
作用：整个程序的入口。在所有可能的模型切分点之间进行遍历，分别计算边缘计算时间、传输时间和云端计算时间，最后选择总时延最小的切分点。
主函数：neuron_surgeon_deployment
输入:param model: DNN 模型层序列或模块列表
    :param network_type: 网络类型，例如 '3g'、'lte' 或 'wifi'
    :param define_speed: 传输带宽或网络速度参数
    :param show: 是否打印每个候选划分点的延迟信息
输出：最佳划分结果res_index

伪代码：
输入：模型层、网络类型和带宽
循环遍历模型层数（所有切分点），如果不是跳过层
首先调用get_input(HW)函数生成图片用作模型推理的输入
然后调用inference_utils文件中的model_partition(model, index)把对模型进行切分。
调用predictor_utils文件中的predict_model_latency函数预测模型在云端和边侧的延迟时间。
调用net_utils文件中的get_speed函数得到带宽并计算传输时间
如果当前总时延更小，更新最佳时延， 更新最佳切分点，遍历结束，输出最佳切分点。

返回最佳切分点编号

get_input(hw)中torch.rand()是PyTorch中生成随机张量(Tensor)的函数。它会生成 0~1之间均匀分布的随机数。
PyTorch里的 Tensor（张量） 可以理解成：
一维 Tensor → 数组
二维 Tensor → 矩阵
三维 Tensor → 图片
四维 Tensor → 一批图片（Batch）
四个数字分别表示：
(Batch, Channel, Height, Width)
1      图片数量
3      RGB三个通道
224    图片高度
224    图片宽度

流程图：

                deployment.py
                       │
                       ▼
        neuron_surgeon_deployment()
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   get_input()   model_partition()   get_layer()
                       │
                ┌──────┴──────┐
                ▼             ▼
      predict_model_latency() edge_model()
                │             │
                ▼             ▼
         Edge Latency     Feature Map
                              │
                              ▼
                        pickle.dumps()
                              │
                              ▼
                        get_speed()
                              │
                              ▼
                  predict_model_latency()
                              │
                              ▼
                      Cloud Latency
                              │
                              ▼
                  Total Latency = Edge + Transmission + Cloud
                              │
                              ▼
                     更新最佳切分点并返回


二、edge_api.py
加载模型；
调用部署算法寻找最佳切分点（昨天已经看懂了 deployment.py）；
按照最佳切分点发起云边协同推理。

程序启动
    │
    ▼
① 测量当前网络带宽
    │
    ▼
② 准备一张输入图片
    │
    ▼
③ 加载神经网络模型
    │
    ▼
④ 根据当前带宽寻找最佳切分点
    │
    ▼
⑤ 按照最佳切分点开始云边协同推理
Neurosurgeon

测一次带宽
        │
        ▼
整个推理都使用这个固定带宽

三、cloud_api.py
云端一直运行，先配合边缘测量网络带宽，然后等待边缘发送模型和数据，完成后继续等待下一次请求。
Cloud启动
↓
while True
↓
启动测速服务器
↓
等待测速完成
↓
创建Socket服务器
↓
等待Edge连接
↓
收到Feature
↓
云端推理
↓
返回结果
↓
继续等待下一次请求



             Edge                             Cloud

启动程序
    │                                   启动程序
    │                                       │
    ├────── 测试带宽 ───────────────────────►│
    │◄──────────────────────────────────────┤
    │
    ├────── deployment寻找最佳切分点
    │
    ├────── start_client() ────────────────► start_server()
    │                                       │
    │     发送模型信息                       │
    │──────────────────────────────────────►│
    │                                       │
    │     发送中间特征                       │
    │──────────────────────────────────────►│
    │                                       │
    │                    云端继续推理
    │◄──────────────────────────────────────│





