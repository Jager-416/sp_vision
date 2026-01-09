#!/bin/bash
# 运行可视化Demo脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cd "$(dirname "$0")/.."

echo "================================"
echo "sp_vision 可视化Demo"
echo "================================"
echo ""

# 检查编译
if [ ! -f "build/auto_aim_test" ]; then
    echo -e "${YELLOW}项目尚未编译，开始编译...${NC}"
    cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    make -C build -j$(nproc)
fi

# 检查demo数据
if [ ! -f "assets/demo/demo.avi" ]; then
    echo "错误: 找不到demo数据文件 assets/demo/demo.avi"
    echo "请确保demo数据已正确放置"
    exit 1
fi

echo -e "${GREEN}可用的Demo:${NC}"
echo "1. auto_aim_test       - 自瞄测试 (离线视频)"
echo "2. auto_buff_test      - 打符测试"
echo "3. camera_test         - 相机测试"
echo "4. detector_video_test - 检测器测试"
echo "5. minimum_vision_system - 最小视觉系统"
echo ""

read -p "请选择Demo编号 (1-5): " choice

case $choice in
    1)
        echo -e "${GREEN}运行自瞄测试...${NC}"
        ./build/auto_aim_test -c configs/demo.yaml assets/demo/demo
        ;;
    2)
        echo -e "${GREEN}运行打符测试...${NC}"
        if [ -f "assets/buff_demo.avi" ]; then
            ./build/auto_buff_test configs/demo.yaml assets/buff_demo
        else
            echo "错误: 找不到buff_demo数据"
        fi
        ;;
    3)
        echo -e "${GREEN}运行相机测试...${NC}"
        echo "请确保相机已连接"
        ./build/camera_test configs/demo.yaml
        ;;
    4)
        echo -e "${GREEN}运行检测器测试...${NC}"
        ./build/detector_video_test configs/demo.yaml assets/demo/demo
        ;;
    5)
        echo -e "${GREEN}运行最小视觉系统...${NC}"
        ./build/minimum_vision_system
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
