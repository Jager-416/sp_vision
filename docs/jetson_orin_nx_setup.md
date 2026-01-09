# Jetson Orin NX 环境配置指南

## 一、系统环境检查

### 1.1 检查JetPack版本

```bash
# 查看JetPack版本
sudo apt-cache show nvidia-jetpack

# 查看CUDA版本
nvcc --version

# 查看L4T版本
cat /etc/nv_tegra_release
```

**推荐版本**:
- JetPack 5.1.2+ (L4T 35.x)
- CUDA 11.4+
- Ubuntu 20.04 or 22.04

---

## 二、基础依赖安装

### 2.1 更新系统

```bash
sudo apt update
sudo apt upgrade -y
```

### 2.2 安装基础开发工具

```bash
sudo apt install -y \
    git \
    g++ \
    cmake \
    build-essential \
    pkg-config \
    can-utils \
    libusb-1.0-0-dev \
    openssh-server \
    screen \
    vim \
    wget \
    curl
```

### 2.3 安装OpenCV (Jetson优化版)

⚠️ **重要**: Jetson已预装OpenCV with CUDA，但可能需要更新

```bash
# 检查已安装的OpenCV版本
python3 -c "import cv2; print(cv2.__version__)"

# 如果版本<4.5，建议手动编译CUDA加速版本
# 或使用JetPack自带的版本
sudo apt install -y libopencv-dev python3-opencv
```

### 2.4 安装C++依赖库

```bash
sudo apt install -y \
    libfmt-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    nlohmann-json3-dev
```

### 2.5 安装Ceres Solver

```bash
# 安装依赖
sudo apt install -y \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev

# 编译安装Ceres
cd ~
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

---

## 三、神经网络推理框架

### ⚠️ 重要决策：OpenVINO vs TensorRT

**当前代码使用**: OpenVINO 2024.6.0
**Jetson最优选择**: TensorRT

#### 方案A: 继续使用OpenVINO (临时方案)

```bash
# 下载OpenVINO for ARM64
cd ~
wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.6/linux/l_openvino_toolkit_ubuntu22_2024.6.0.16997.0d1e491efb6_arm64.tgz

# 解压
tar -xvzf l_openvino_toolkit_ubuntu22_2024.6.0.16997.0d1e491efb6_arm64.tgz
sudo mv l_openvino_toolkit_ubuntu22_2024.6.0.16997.0d1e491efb6_arm64 /opt/intel/openvino_2024.6.0

# 设置环境变量
echo "source /opt/intel/openvino_2024.6.0/setupvars.sh" >> ~/.bashrc
source ~/.bashrc

# 安装依赖
cd /opt/intel/openvino_2024.6.0/install_dependencies
sudo -E ./install_openvino_dependencies.sh
```

**缺点**:
- 无法充分利用Jetson GPU
- 性能只有TensorRT的30-40%
- 仅建议用于快速验证

#### 方案B: 迁移到TensorRT (强烈推荐)

TensorRT已随JetPack预装，无需额外安装。

```bash
# 验证TensorRT安装
dpkg -l | grep TensorRT
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 通常位于
ls /usr/src/tensorrt/
```

**优点**:
- 性能提升2-3倍
- NVIDIA官方支持Jetson
- 低延迟GPU推理

**注意**: 需要修改代码（稍后提供迁移步骤）

---

## 四、相机驱动安装

### 4.1 海康机器视觉相机

```bash
cd ~
wget https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_GML_V2.1.2_231116.zip
unzip MVS_STD_GML_V2.1.2_231116.zip
cd MVS_STD_GML_V2.1.2_231116

# 安装ARM64版本
cd arm64
sudo dpkg -i *.deb
```

### 4.2 迈德威视相机 (备选)

```bash
# 从官网下载ARM版本SDK
# https://mindvision.com.cn/category/software/sdk-installation-package/
# 选择 Linux ARM64 版本

cd ~/Downloads
tar -xvf linuxSDK_V*.tar.gz
cd linuxSDK
sudo ./install.sh
```

---

## 五、编译sp_vision项目

### 5.1 克隆项目 (如果还没有)

```bash
cd ~
git clone <your-repo-url> sp_vision
cd sp_vision
git checkout claude/openvino-code-review-AAAXx
```

### 5.2 修改CMakeLists.txt (Jetson适配)

编辑 `CMakeLists.txt`，修改第15行：

```cmake
# 原来：
# set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")

# 改为Jetson路径：
set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake/")
```

### 5.3 编译

```bash
cd ~/sp_vision

# 生成compile_commands.json (用于clangd)
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ln -sf build/compile_commands.json .

# 编译 (使用所有CPU核心)
make -C build -j$(nproc)
```

**预计编译时间**: 5-10分钟

---

## 六、运行可视化Demo

### 6.1 准备测试数据

下载demo视频和配置文件：

```bash
# 项目自带demo数据
ls assets/demo/

# 应该包含：
# - demo.avi (视频文件)
# - demo.txt (IMU四元数数据)
```

### 6.2 运行自瞄测试 (离线视频)

```bash
cd ~/sp_vision

# 运行auto_aim_test
./build/auto_aim_test \
    -c configs/demo.yaml \
    assets/demo/demo
```

**期望输出**:
- OpenCV窗口显示装甲板检测结果
- 终端输出FPS和延迟信息
- 绘制瞄准点、预测轨迹等

**可视化界面**:
- 绿色框: YOLO检测到的装甲板
- 红色点: 瞄准点
- 白色文字: 云台角度、控制指令

**操作**:
- `q`: 退出
- `空格`: 暂停/继续

### 6.3 运行多线程检测测试

```bash
# 需要连接相机和CAN设备
./build/mt_auto_aim_debug configs/sentry.yaml
```

**前提条件**:
- 海康/迈德威视相机已连接
- CAN总线已配置 (或使用串口)
- 配置文件中的设备路径正确

### 6.4 其他可用的Demo

#### a) 相机测试
```bash
./build/camera_test configs/demo.yaml
```

#### b) USB相机测试
```bash
./build/usbcamera_test
```

#### c) 打符测试
```bash
./build/auto_buff_test configs/demo.yaml assets/buff_demo
```

#### d) 最小视觉系统
```bash
./build/minimum_vision_system
```

---

## 七、串口/CAN配置

### 7.1 USB串口权限

```bash
# 添加当前用户到dialout组
sudo usermod -a -G dialout $USER

# 重新登录或运行
newgrp dialout

# 验证
ls -l /dev/ttyACM*
```

### 7.2 创建串口udev规则

```bash
# 查看设备信息
udevadm info -a -n /dev/ttyACM0 | grep -E 'serial|idVendor|idProduct'

# 创建规则文件
sudo nano /etc/udev/rules.d/99-usb-serial.rules
```

写入 (替换为实际的idVendor和idProduct):
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", SYMLINK+="stm32_usb", MODE="0666"
```

```bash
# 重新加载规则
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 7.3 CAN设置 (如果使用USB2CAN)

```bash
# 安装can-utils
sudo apt install can-utils

# 启动CAN接口
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

# 验证
candump can0
```

---

## 八、性能优化 (Jetson专用)

### 8.1 设置最大性能模式

```bash
# 查看当前功耗模式
sudo nvpmodel -q

# 设置最大性能模式 (MAXN)
sudo nvpmodel -m 0

# 锁定最高频率
sudo jetson_clocks
```

### 8.2 启用风扇

```bash
# 设置风扇为100%
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

### 8.3 监控性能

```bash
# 实时监控GPU/CPU/温度
sudo tegrastats

# 或使用jtop (需要安装)
sudo pip3 install jetson-stats
sudo jtop
```

---

## 九、远程桌面设置

### 9.1 安装NoMachine

```bash
cd ~/Downloads

# 下载ARM64版本
wget https://download.nomachine.com/download/8.11/Arm/nomachine_8.11.3_1_arm64.deb

# 安装
sudo dpkg -i nomachine_8.11.3_1_arm64.deb

# 启动服务
sudo systemctl enable nxserver
sudo systemctl start nxserver
```

### 9.2 从PC连接

1. 在PC安装NoMachine客户端
2. 输入Jetson的IP地址
3. 使用用户名/密码登录
4. 可以运行带GUI的demo

---

## 十、常见问题排查

### 10.1 OpenVINO找不到

**错误**: `Could not find OpenVINO`

**解决**:
```bash
# 确认安装路径
ls /opt/intel/openvino_2024.6.0/

# 手动设置环境变量
export OpenVINO_DIR=/opt/intel/openvino_2024.6.0/runtime/cmake/
```

### 10.2 相机无法打开

**错误**: `Failed to open camera`

**解决**:
```bash
# 检查USB设备
lsusb

# 检查相机权限
ls -l /dev/bus/usb/

# 测试相机
./build/camera_test
```

### 10.3 CUDA内存不足

**错误**: `out of memory`

**解决**:
```bash
# 检查GPU内存
tegrastats

# 减小batch size或模型输入尺寸
# 修改configs/*.yaml中的相关参数
```

### 10.4 FPS过低

**可能原因**:
1. 未开启最大性能模式 → `sudo nvpmodel -m 0`
2. 使用OpenVINO而非TensorRT → 考虑迁移
3. 相机分辨率过高 → 降低到1280x1024或更低

---

## 十一、下一步：迁移到TensorRT

为了获得最佳性能，建议迁移到TensorRT。主要步骤：

1. 导出YOLO模型为ONNX格式
2. 使用trtexec转换为TRT引擎
3. 修改推理代码使用TensorRT API
4. 测试性能提升

**预期性能提升**: 从60-80 FPS → 150-250 FPS

---

## 附录A: 推荐配置

### Jetson Orin NX最佳配置

| 项目 | 推荐值 |
|------|--------|
| 功耗模式 | MAXN (25W) |
| 相机分辨率 | 1280x1024 @ 120fps |
| 模型 | YOLOV8s (TensorRT FP16) |
| 推理设备 | GPU (TensorRT) |
| 通信方式 | USB串口 (CDC) |

### 性能基准 (TensorRT)

- YOLOV8s推理: 4-6ms (GPU)
- 端到端延迟: 8-12ms
- 系统FPS: 100-150 fps
- GPU利用率: 70-85%
- 功耗: 18-22W

---

## 附录B: 配置文件示例

创建 `configs/jetson_orin.yaml`:

```yaml
# 相机配置
camera_name: "hikrobot"
camera_index: 0
resolution_width: 1280
resolution_height: 1024
fps: 120

# YOLO配置
yolo_name: "yolov8"
yolov8_model_path: "models/yolov8s.onnx"  # 或 .trt
device: "GPU"  # OpenVINO: GPU/CPU, TensorRT: CUDA:0

# 串口配置
com_port: "/dev/ttyACM0"

# 性能配置
use_roi: true
roi_size: 640
debug: true
```

---

## 结束

完成以上步骤后，你的Jetson Orin NX应该能够运行sp_vision项目了。

如有问题，请检查：
1. 环境变量是否设置正确
2. 所有依赖是否安装完成
3. 设备权限是否配置
4. 配置文件路径是否正确

祝调试顺利！🚀
