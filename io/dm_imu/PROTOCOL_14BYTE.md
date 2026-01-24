# 14字节自定义IMU协议适配

## 协议发现

通过数据分析发现,你的下位机使用的是**14字节自定义协议**,而不是ROS协议或DM协议。

### 协议格式

```
偏移  字节数  类型    含义
0-3   4      float   yaw (弧度)
4-7   4      float   pitch (弧度)
8-11  4      float   roll (弧度)
12-13 2      uint16  填充/同步标记 (固定为 0x0000)
```

**总包长**: 14字节
**字节序**: 小端序 (little-endian)
**数据率**: 约100-200 Hz (根据你的数据推测)

### 示例数据

```
23 9e 20 c0 3b af d5 bd 8e 62 fc bc 00 00
^^^^^^^^^^^  yaw=-2.509652 rad = -143.80°
            ^^^^^^^^^^^ pitch=-0.104338 rad = -5.98°
                        ^^^^^^^^^^^ roll=-0.030809 rad = -1.77°
                                    ^^^^^ 同步标记 0x0000
```

## 主要修改

### 1. 数据结构简化

```cpp
struct __attribute__((packed)) IMU_Packet
{
  float yaw;      // 0-3
  float pitch;    // 4-7
  float roll;     // 8-11
  uint16_t padding; // 12-13: 总是 0x0000
};
```

### 2. 同步机制

使用`0x00 0x00`作为包尾标识进行同步:

```cpp
// 查找包尾标记
if (*(it + 12) == 0x00 && *(it + 13) == 0x00) {
  found_sync = true;
  packet_start = it;
  break;
}
```

### 3. 数据验证

添加了合理性检查:

```cpp
// 检查NaN和范围
if (std::isnan(yaw_rad) || std::isnan(pitch_rad) || std::isnan(roll_rad) ||
    std::abs(yaw_rad) > 10.0f || std::abs(pitch_rad) > 10.0f || 
    std::abs(roll_rad) > 10.0f) {
  // 跳过无效数据
}
```

### 4. 欧拉角到四元数转换

```cpp
// ZYX旋转顺序
Eigen::Quaterniond q = 
  Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()) *
  Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
  Eigen::AngleAxisd(roll_rad, Eigen::Vector3d::UnitX());
```

## 使用方法

### 1. 替换文件

```bash
cp dm_imu_14byte.hpp io/dm_imu/dm_imu.hpp
cp dm_imu_14byte.cpp io/dm_imu/dm_imu.cpp
```

### 2. 编译测试

```bash
cd io
make clean
make
cd ..
./dm_test
```

### 3. 预期输出

```
[2026-01-25 xx:xx:xx.xxx] [info] [DM_IMU] serial port opened
[2026-01-25 xx:xx:xx.xxx] [info] [DM_IMU] Waiting for initial IMU data...
[2026-01-25 xx:xx:xx.xxx] [info] [DM_IMU] initialized
[2026-01-25 xx:xx:xx.xxx] [info] [DM_IMU] Packets: 123, yaw: -143.80°, pitch: -5.98°, roll: -1.77°
[2026-01-25 xx:xx:xx.xxx] [info] z-143.80 y-5.98 x-1.77 degree
```

## 性能特征

- **延迟**: 极低 (无CRC校验,直接解析)
- **可靠性**: 依赖`0x0000`同步,可能出现误同步
- **数据率**: 取决于下位机发送频率

## 潜在问题

### 问题1: 误同步

如果float数据中恰好出现`0x00 0x00`,可能导致误同步。

**解决方案A** (推荐): 联系下位机团队添加包头
```cpp
// 建议的改进协议:
// 0xAA 0x55 [yaw][pitch][roll] 0x00 0x00
```

**解决方案B**: 添加数值范围检查(已实现)

### 问题2: 字节序

当前假设小端序。如果数据异常,尝试大端序:

```cpp
// 在IMU_Packet定义前添加
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#warning "Big endian system detected"
#endif
```

### 问题3: send()未实现

当前协议只有接收,没有发送格式。如果需要发送:

1. 确认下位机的接收协议
2. 实现对应的send()函数

## 协议对比

| 特性 | 14字节协议 | ROS协议 | DM协议 |
|------|-----------|---------|--------|
| 包头 | 无 | 0xA5 | 0x55 0xAA |
| 长度 | 固定14字节 | 动态 | 固定57字节 |
| CRC | 无 | CRC8+CRC16 | CRC16 |
| 数据 | 欧拉角 | 欧拉角+标志位 | 加速度+角速度+欧拉角 |
| 可靠性 | 低 | 高 | 中 |
| 解析复杂度 | 低 | 高 | 中 |

## 调试技巧

### 查看原始包

添加hexdump输出:

```cpp
tools::logger()->debug("[DM_IMU] Raw packet: {:02x} {:02x} {:02x} ...", 
                      packet->yaw, packet->pitch, packet->roll);
```

### 监控同步状态

```cpp
static int sync_errors = 0;
if (found_sync) {
  sync_errors = 0;
} else {
  sync_errors++;
  if (sync_errors > 100) {
    tools::logger()->warn("[DM_IMU] Lost sync!");
  }
}
```

### 验证数据连续性

```cpp
static float last_yaw = 0;
float delta = std::abs(yaw_rad - last_yaw);
if (delta > 0.5) {  // 跳变超过0.5弧度
  tools::logger()->warn("[DM_IMU] Large yaw jump: {:.2f}", delta);
}
last_yaw = yaw_rad;
```

## 常见问题

**Q: 为什么没有包头?**
A: 这可能是简化的调试协议,或者硬件团队的快速实现。建议联系他们添加标准包头。

**Q: 数据跳变怎么办?**
A: 添加卡尔曼滤波或互补滤波平滑数据。

**Q: 如何提高可靠性?**
A: 建议下位机升级协议,添加:
- 包头 (0xAA 0x55)
- 包序号
- CRC16校验
- 包尾 (0x0D 0x0A)

## 后续优化

1. **添加卡尔曼滤波**: 平滑角度数据
2. **实现send()**: 如果需要双向通信
3. **添加时间戳**: 在包中包含下位机时间戳
4. **多传感器融合**: 如果有其他IMU数据源

## 联系下位机团队

建议询问:
1. 完整的协议文档
2. 是否计划升级协议
3. 发送命令的协议格式
4. 数据发送频率
