import pickle
import torch
from utils import inference_utils
from predictor import predictor_utils
from net import net_utils

# deployment.py
# 负责根据给定的DNN模型、网络类型和带宽，计算最优的模型切分点（partition point），
# 以便在边缘设备和云端之间协同推理，实现最低总延迟。


def get_layer(model, point):
    """
    返回模型切分点对应的上一层。

    :param model: 可迭代的模型层列表
    :param point: 切分点索引，0 表示未切分
    :return: 切分点前的层对象，point==0 时返回 None
    """
    if point == 0:
        layer = None #未切分
    else:
        layer = model[point - 1] #返回模型的上一层
    return layer


def get_input(HW):
    """
    生成一个用于延迟预测的随机输入张量。

    :param HW: 输入图像的高度和宽度
    :return: 形状为 (1, 3, HW, HW) 的随机 torch.Tensor
    """
    #生成一张大小为 HW×HW 的随机 RGB 图片，用作模型推理的输入，不需要计算梯度。
    return torch.rand(size=(1, 3, HW, HW), requires_grad=False)


def neuron_surgeon_deployment(model, network_type, define_speed, show=False):
    """
    通过遍历每个可能的划分点，计算边缘计算、传输和云端推理的总延迟，
    选取总延迟最小的划分点作为最终部署策略。

    :param model: DNN 模型层序列或模块列表
    :param network_type: 网络类型，例如 '3g'、'lte' 或 'wifi'
    :param define_speed: 传输带宽或网络速度参数
    :param show: 是否打印每个候选划分点的延迟信息
    :return: 最优划分点索引
    """
    res_lat = None
    res_index = None
    res_layer_index = None
    predictor_dict = {}

    layer_index = 0  # 仅用于记录非跳过层的顺序索引
    for index in range(len(model) + 1):
        # 如果当前层是跳过层，则不计算该位置的划分点
        if index != 0 and predictor_utils.skip_layer(model[index - 1]):
            continue

        # 使用固定输入尺寸生成输入张量，用于延迟预测
        x = get_input(HW=224)

        # 将模型按 index 划分为 edge_model 和 cloud_model
        # index == 0: edge_model 为空，全部在云端运行
        # index == len(model): cloud_model 为空，全部在边缘运行
        edge_model, cloud_model = inference_utils.model_partition(model, index)

        # 在边缘侧预测子模型的执行时间
        edge_lat = predictor_utils.predict_model_latency(
            x, edge_model, device="edge", predictor_dict=predictor_dict)
        x = edge_model(x)

        # 计算数据传输时间，基于输出激活的序列化大小和网络带宽
        transport_size = len(pickle.dumps(x))
        speed = net_utils.get_speed(network_type=network_type, bandwidth=define_speed)
        transmission_lat = transport_size / speed

        # 在云端预测剩余子模型的执行时间
        cloud_lat = predictor_utils.predict_model_latency(
            x, cloud_model, device="cloud", predictor_dict=predictor_dict)

        # 总延迟为边缘执行 + 传输 + 云端执行
        total_lat = edge_lat + transmission_lat + cloud_lat

        now_layer = get_layer(model, index)
        if show:
            print(
                f"index {layer_index + 1} - layer : {now_layer} \n"
                f"edge latency : {edge_lat:.2f} ms , transmit latency : {transmission_lat:.2f} ms , "
                f"cloud latency : {cloud_lat:.2f} ms , total latency : {total_lat:.2f} ms"
            )
            print("----------------------------------------------------------------------------------------------------------")

        # 保存当前最优划分点
        if res_lat is None or total_lat < res_lat:
            res_lat = total_lat
            res_index = index
            res_layer_index = layer_index
        layer_index += 1

    # 输出最佳划分点结果
    res_layer = get_layer(model, res_index)
    print(f"best latency : {res_lat:.2f} ms , best partition point : {res_layer_index} - {res_layer}")
    print("----------------------------------------------------------------------------------------------------------")

    return res_index




