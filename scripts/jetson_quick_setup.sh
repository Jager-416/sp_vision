#!/bin/bash
# Jetson Orin NX 快速配置脚本
# 用途: 自动安装sp_vision所需的基础依赖

set -e  # 遇到错误立即退出

echo "================================"
echo "Jetson Orin NX 快速配置脚本"
echo "================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在Jetson设备上运行
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${RED}错误: 此脚本仅适用于Jetson设备${NC}"
    exit 1
fi

echo -e "${GREEN}检测到Jetson设备:${NC}"
cat /etc/nv_tegra_release
echo ""

# 检查JetPack版本
echo -e "${YELLOW}[1/6] 检查JetPack版本...${NC}"
dpkg -l | grep nvidia-jetpack || echo "无法检测到nvidia-jetpack包"
echo ""

# 更新系统
echo -e "${YELLOW}[2/6] 更新系统...${NC}"
sudo apt update
echo ""

# 安装基础依赖
echo -e "${YELLOW}[3/6] 安装基础开发工具...${NC}"
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

echo ""

# 安装C++库
echo -e "${YELLOW}[4/6] 安装C++依赖库...${NC}"
sudo apt install -y \
    libopencv-dev \
    libfmt-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    nlohmann-json3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev

echo ""

# 配置串口权限
echo -e "${YELLOW}[5/6] 配置串口权限...${NC}"
sudo usermod -a -G dialout $USER
echo -e "${GREEN}已添加$USER到dialout组，需要重新登录才能生效${NC}"
echo ""

# 设置最大性能模式
echo -e "${YELLOW}[6/6] 设置Jetson最大性能模式...${NC}"
read -p "是否设置为最大性能模式MAXN? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo nvpmodel -m 0
    sudo jetson_clocks
    echo -e "${GREEN}已设置为最大性能模式${NC}"
else
    echo "跳过性能模式设置"
fi
echo ""

# 检查TensorRT
echo -e "${YELLOW}检查TensorRT安装...${NC}"
if dpkg -l | grep -q tensorrt; then
    echo -e "${GREEN}TensorRT已安装:${NC}"
    dpkg -l | grep tensorrt | head -3
else
    echo -e "${RED}未检测到TensorRT，请确保JetPack已正确安装${NC}"
fi
echo ""

# 检查CUDA
echo -e "${YELLOW}检查CUDA安装...${NC}"
if command -v nvcc &> /dev/null; then
    echo -e "${GREEN}CUDA已安装:${NC}"
    nvcc --version | grep release
else
    echo -e "${RED}未检测到CUDA${NC}"
fi
echo ""

echo "================================"
echo -e "${GREEN}基础依赖安装完成!${NC}"
echo "================================"
echo ""
echo "下一步:"
echo "1. 重新登录以应用串口权限 (或运行: newgrp dialout)"
echo "2. 安装Ceres Solver (运行: ./scripts/install_ceres.sh)"
echo "3. 安装相机SDK (海康或迈德威视)"
echo "4. 选择OpenVINO或TensorRT作为推理后端"
echo "5. 编译项目: cmake -B build && make -C build -j\$(nproc)"
echo ""
echo -e "${YELLOW}提示: 查看详细文档 docs/jetson_orin_nx_setup.md${NC}"
