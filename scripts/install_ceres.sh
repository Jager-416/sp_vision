#!/bin/bash
# Ceres Solver 安装脚本 (Jetson优化版)

set -e

echo "================================"
echo "安装 Ceres Solver"
echo "================================"

# 检查依赖
echo "[1/4] 检查依赖..."
sudo apt install -y \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev

# 下载源码
echo "[2/4] 下载Ceres源码..."
cd /tmp
if [ -d "ceres-solver" ]; then
    rm -rf ceres-solver
fi
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver

# 编译
echo "[3/4] 编译Ceres (这可能需要10-15分钟)..."
mkdir build
cd build
cmake ..
make -j$(nproc)

# 安装
echo "[4/4] 安装Ceres..."
sudo make install

echo ""
echo "================================"
echo "Ceres Solver 安装完成!"
echo "================================"
echo "版本信息:"
pkg-config --modversion ceres
echo ""
