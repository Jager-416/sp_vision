#!/usr/bin/env python3
import serial
import struct
import sys

def find_imu_packets(data):
    """在数据中查找可能的IMU包"""
    print(f"\n=== Analyzing {len(data)} bytes ===\n")
    
    # 方法1: 查找末尾为00 00的14字节包
    print("Method 1: Looking for 14-byte packets ending with 00 00")
    found_count = 0
    for i in range(len(data) - 14):
        if data[i+12] == 0 and data[i+13] == 0:
            # 检查不是全0
            if any(data[i:i+12]):
                packet = data[i:i+14]
                yaw, pitch, roll = struct.unpack('<fff', packet[0:12])
                
                # 检查是否是合理的角度值
                if abs(yaw) < 10 and abs(pitch) < 10 and abs(roll) < 10:
                    if not (yaw == 0 and pitch == 0 and roll == 0):
                        found_count += 1
                        if found_count <= 5:
                            print(f"  [@{i:04d}] {packet.hex(' ')}")
                            print(f"         yaw={yaw:7.3f} pitch={pitch:7.3f} roll={roll:7.3f}")
                            print(f"         yaw={yaw*57.3:7.1f}° pitch={pitch*57.3:7.1f}° roll={roll*57.3:7.1f}°")
    
    print(f"Found {found_count} valid packets\n")
    
    # 方法2: 统计字节模式
    print("Method 2: Byte patterns (most common 2-byte sequences)")
    from collections import Counter
    pairs = Counter()
    for i in range(len(data)-1):
        pairs[data[i:i+2].hex()] += 1
    
    for pattern, count in pairs.most_common(10):
        print(f"  {pattern}: {count} times")
    
    # 方法3: 查找重复间隔
    print("\nMethod 3: Looking for repeating intervals")
    zeros = [i for i in range(len(data)-1) if data[i] == 0 and data[i+1] == 0]
    if len(zeros) > 1:
        intervals = [zeros[i+1] - zeros[i] for i in range(min(20, len(zeros)-1))]
        print(f"  First 20 intervals between '00 00': {intervals}")
        most_common = Counter(intervals).most_common(1)[0]
        print(f"  Most common interval: {most_common[0]} bytes ({most_common[1]} times)")

port = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyACM0'
baud = int(sys.argv[2]) if len(sys.argv) > 2 else 921600

print(f"Reading from {port} at {baud} baud...")
ser = serial.Serial(port, baud, timeout=1)

import time
time.sleep(0.5)

data = bytearray()
for _ in range(50):
    if ser.in_waiting:
        data.extend(ser.read(ser.in_waiting))
    time.sleep(0.01)

ser.close()

if len(data) == 0:
    print("ERROR: No data received!")
    sys.exit(1)

find_imu_packets(data)

# 保存到文件
filename = f'imu_dump_{int(time.time())}.bin'
with open(filename, 'wb') as f:
    f.write(data)
print(f"\nData saved to {filename}")
