一、get_datasets_func.py（数据/训练来源）
👉 目标：

latency regression 数据怎么来的？

你要搞清楚：

feature是什么
label是什么
是真实测量还是模拟


get_datasets_by_kernel_kind
"""
    根据传入的kernel_kind - 返回对应的处理函数
    :param kernel_kind: kernel_kind : 0 - conv2d , 1 - dw_conv2d , 2 - linear, 3 - maxPool2d , 4 - avgPool2d, 5 - BatchNorm
    :return: features,targets
    """

get_datasets_func.py 的所有函数都输出固定的二元组 (data_features, data_targets)：

data_features：pandas.DataFrame，每一行对应一个 DNN 层的特征（如 HW、Cout 等），作为模型输入 X。
data_targets：pandas.Series，每一行对应这层在真实设备上测得的推理延迟 latency，作为模型训练标签 y。

这两个对象会直接传给 model_training_linear()，用于训练后续的延迟预测器。


二、 net_utils.py（网络通信 & 带宽模拟）

👉 目标：

bandwidth怎么测 + socket怎么传tensor

这个 `net_utils.py` 文件是一个专门用于深度神经网络端-云协同推理（DNN Edge-Cloud Collaborative Inference）的网络通信底层工具类。

它的核心思想是：将一个庞大的深度学习模型**拆分为两部分**。前半部分在计算能力较弱的“边缘端（Client）”运行，运行到指定的切分点（Partition Point）后，将产生的中间层特征数据（Tensors）通过网络发送给计算能力更强的“云端（Server）”，由云端完成剩余的后半部分推理，并返回最终结果或时延统计。

以下是对该文件核心功能模块的详细拆解：

---

### 1. 核心业务流程（协同推理）

文件开头的两个核心函数定义了客户端（边缘端）和服务器（云端）之间握手、传输、计算的完整生命周期：

* **`start_client(...)`（边缘端主控）：**
1. 建立与云端服务器的 TCP 连接。
2. 告知云端本次实验所使用的**模型类型**（`model_type`）和**网络切分点**（`partition_point`）。
3. 加载并切分模型，在本地（CPU 或 GPU）运行前半部分模型，并使用 `inference_utils.recordTime` 记录边缘端推理时延。
4. 通过网络将中间层输出（`edge_output`）发送给云端。
5. 同步等待并接收云端返回的传输时延和云端推理时延。


* **`start_server(...)`（云端主控）：**
1. 监听并接收客户端的连接请求。
2. 接收模型类型和切分点，并在云端动态构建出后半部分模型，加载到指定设备（如 CUDA）。
3. 接收边缘端传过来的中间层数据，同时记录网络传输所消耗的时间。
4. 在云端运行后半部分模型，并记录云端推理时延。
5. 将测得的时延数据打包发回客户端，完成一次协同推理。



---

### 2. 自定义数据传输协议（大体积 Tensor 传输）

由于深度学习的中间层特征图（Tensor）体积通常较大，如果直接通过普通的 `socket.send` 发送，在 TCP 协议下极易产生**粘包**或**分片断包**问题。文件中设计了两种传输流：

#### A. 短数据传输（Metadata）

* **函数：** `send_short_data` / `get_short_data`
* **逻辑：** 针对模型名称、切分点索引（整数）、时延（浮点数）等极其简短的数据，直接使用 `pickle.dumps` 序列化成字节流并一次性发送/接收（限制在 1024 字节内）。

#### B. 长数据传输（Tensors/Feature Maps）

* **函数：** `send_data` / `get_data`
* **逻辑：** 采用“长度预报 + 循环接收”的经典 TCP 可靠传输方案：
1. **发送方**先将大对象（如 Tensor）通过 `pickle` 序列化，计算出总字节长度。
2. **发送方**先发送这个长度给接收方，接收方收到后回复 `"yes len"`（确认握手）。
3. **发送方**开始发送真实的原始数据，**接收方**进入 `while True` 循环，每次最多读取 40960 字节，直到接收到的累计字节数 $\ge$ 预报的长度。
4. 在循环接收的过程中，利用 `time.perf_counter()` 动态统计每一次网络 `recv` 的耗时，从而精准计算出**网络传输时延**。



> **💡 关于代码中的“防粘包”处理：**
> 在 `start_client` 和 `start_server` 中，能看到多处 `conn.recv(40)` 和 `conn.sendall("avoid sticky".encode())`。这是因为连续快速调用 `send_short_data` 时，TCP 底层的 Nagle 算法可能会把多个小包合并发送。这种设计相当于增加了一个“同步屏障（Barrier）”，强迫两端对齐步调，防止数据串线。

---

### 3. 套接字与生命周期管理

这部分是对标准 Python `socket` 库的面向对象封装：

* **`get_socket_server(...)`：** 负责初始化云端的监听套接字。内部做了**跨平台兼容**：
* 如果是 **Windows** 系统，设置 `SO_REUSEADDR` 允许端口重用。
* 如果是 **Linux / MacOS** 系统，设置 `SO_REUSEPORT` 允许多个套接字绑定到同一端口（提高并发性能）。


* **`get_socket_client(...)`：** 边缘端创建套接字并主动向云端 IP 和端口发起 `connect`。
* **`close_conn` / `close_socket` / `wait_client`：** 标准的关闭连接、关闭服务以及阻塞等待客户端接入的函数。

---

### 4. 网络带宽与时延预测（实验辅助）

为了分析协同推理在不同网络环境下的表现，文件还集成了网络测速与理论预测模块：

* **`get_bandwidth()`：** 利用 `speedtest-cli` 库（代码中简写为 `spt`）实时测试当前节点的公网带宽（主要是上传速度 `upload`），用于评估网络环境。
* **`get_speed(...)`：** 将常见的网络类型（3G、LTE、WiFi）以及带宽大小，换算成每毫秒可以传输的字节数（$Bytes/ms$），方便在算法中进行理论数学建模。
* **`create_server` 和 `show_speed`：** 纯测试函数。通过计算 `数据大小 / 实际时延` 得到实际传输速度，并将其与专业测速工具（如 iperf）的理论预测值进行打印对比，用以验证时延统计代码的准确性。

---

### 总结

这个文件是端云协同推理实验的**交通枢纽**。它不仅解决了深度学习对象（Tensor）在网络上的可靠序列化传输问题，还精准地实现了“边缘计算 -> 网络传输 -> 云端计算”三阶段的时间轴切片与统计。你的主程序（如 `edge_api.py` 和 `cloud_api.py`）只需调用里面的 `start_client` 和 `start_server`，就能跑通一整套协同推理流。


