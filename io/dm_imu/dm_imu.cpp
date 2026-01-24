#include "dm_imu.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace io
{

struct __attribute__((packed)) IMU_Packet_14Byte
{
  float yaw;
  float pitch;
  float roll;
  uint16_t padding;
};

DM_IMU::DM_IMU() 
: bullet_speed(0),
  mode(Mode::idle),
  shoot_mode(ShootMode::left_shoot),
  ft_angle(0),
  queue_(5000)
{
  init_serial();
  rec_thread_ = std::thread(&DM_IMU::get_imu_data_thread, this);
  queue_.pop(data_ahead_);
  queue_.pop(data_behind_);
  tools::logger()->info("[DM_IMU] initialized");
}

DM_IMU::~DM_IMU()
{
  stop_thread_ = true;
  if (rec_thread_.joinable()) {
    rec_thread_.join();
  }
  if (serial_.isOpen()) {
    serial_.close();
  }
}

void DM_IMU::init_serial()
{
  try {
    serial_.setPort("/dev/ttyACM0");
    serial_.setBaudrate(921600);
    serial_.setFlowcontrol(serial::flowcontrol_none);
    serial_.setParity(serial::parity_none);
    serial_.setStopbits(serial::stopbits_one);
    serial_.setBytesize(serial::eightbits);
    serial::Timeout time_out = serial::Timeout::simpleTimeout(20);
    serial_.setTimeout(time_out);
    serial_.open();
    usleep(1000000);

    tools::logger()->info("[DM_IMU] serial port opened");
  }
  catch (serial::IOException & e) {
    tools::logger()->warn("[DM_IMU] failed to open serial port ");
    exit(0);
  }
}

void DM_IMU::get_imu_data_thread()
{
  std::vector<uint8_t> buffer;
  buffer.reserve(16384);
  bool synced = false;
  size_t sync_offset = 0;
  
  while (!stop_thread_) {
    if (!serial_.isOpen()) {
      tools::logger()->warn("[DM_IMU] serial port closed");
      usleep(100000);
      continue;
    }

    try {
      std::vector<uint8_t> chunk(64);
      size_t bytes_read = serial_.read(chunk.data(), 64);
      
      if (bytes_read == 0) {
        continue;
      }
      
      buffer.insert(buffer.end(), chunk.begin(), chunk.begin() + bytes_read);
      
      if (buffer.size() > 16384) {
        if (synced && sync_offset < buffer.size()) {
          std::vector<uint8_t> tmp(buffer.begin() + sync_offset, buffer.end());
          buffer = std::move(tmp);
          sync_offset = 0;
        } else {
          buffer.clear();
          synced = false;
          sync_offset = 0;
        }
      }
      
      if (!synced) {
        for (size_t i = 0; i + 13 < buffer.size(); ++i) {
          if (buffer[i + 12] == 0x00 && buffer[i + 13] == 0x00) {
            sync_offset = i;
            synced = true;
            tools::logger()->info("[DM_IMU] Synced at buffer offset {}", i);
            break;
          }
        }
        
        if (!synced) {
          if (buffer.size() > 14) {
            buffer.erase(buffer.begin(), buffer.end() - 13);
          }
          continue;
        }
      }
      
      while (synced && sync_offset + 14 <= buffer.size()) {
        if (buffer[sync_offset + 12] != 0x00 || buffer[sync_offset + 13] != 0x00) {
          // tools::logger()->warn("[DM_IMU] Lost sync at offset {}, resynchronizing", sync_offset);
          synced = false;
          buffer.erase(buffer.begin(), buffer.begin() + sync_offset + 1);
          sync_offset = 0;
          break;
        }
        
        IMU_Packet_14Byte* packet = reinterpret_cast<IMU_Packet_14Byte*>(&buffer[sync_offset]);
        
        float yaw_rad = packet->yaw;
        float pitch_rad = packet->pitch;
        float roll_rad = packet->roll;
        
        if (std::isnan(yaw_rad) || std::isnan(pitch_rad) || std::isnan(roll_rad) ||
            std::abs(yaw_rad) > 10.0f || std::abs(pitch_rad) > 10.0f || 
            std::abs(roll_rad) > 10.0f) {
          tools::logger()->debug(
            "[DM_IMU] Invalid data: yaw={:.2f}, pitch={:.2f}, roll={:.2f}",
            yaw_rad, pitch_rad, roll_rad);
          sync_offset += 14;
          continue;
        }
        
        data.yaw = yaw_rad * 180.0f / M_PI;
        data.pitch = pitch_rad * 180.0f / M_PI;
        data.roll = roll_rad * 180.0f / M_PI;
        
        auto timestamp = std::chrono::steady_clock::now();
        Eigen::Quaterniond q = Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(roll_rad, Eigen::Vector3d::UnitX());
        q.normalize();
        queue_.push({q, timestamp});
        
        tools::logger()->debug(
          "[DM_IMU] IMU data: yaw={:.2f}°, pitch={:.2f}°, roll={:.2f}°",
          data.yaw, data.pitch, data.roll);
        
        sync_offset += 14;
      }
      
      if (synced && sync_offset > 0) {
        buffer.erase(buffer.begin(), buffer.begin() + sync_offset);
        sync_offset = 0;
      }
      
    } catch (const std::exception& e) {
      tools::logger()->error("[DM_IMU] Error in receive thread: {}", e.what());
      usleep(100000);
    }
  }
}

Eigen::Quaterniond DM_IMU::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  if (data_behind_.timestamp < timestamp) data_ahead_ = data_behind_;

  while (true) {
    queue_.pop(data_behind_);
    if (data_behind_.timestamp > timestamp) break;
    data_ahead_ = data_behind_;
  }

  Eigen::Quaterniond q_a = data_ahead_.q.normalized();
  Eigen::Quaterniond q_b = data_behind_.q.normalized();
  auto t_a = data_ahead_.timestamp;
  auto t_b = data_behind_.timestamp;
  auto t_c = timestamp;
  std::chrono::duration<double> t_ab = t_b - t_a;
  std::chrono::duration<double> t_ac = t_c - t_a;

  auto k = t_ac / t_ab;
  Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();

  return q_c;
}

void DM_IMU::send(Command command)
{
  Command_Send_Frame frame;
  frame.FrameHeader = 0x55;
  frame.flag = 0xBB;
  frame.slave_id = 0x01;
  frame.reg_command = 0x04;
  frame.control = command.control ? 1 : 0;
  frame.shoot = command.shoot ? 1 : 0;
  frame.yaw = (int16_t)(command.yaw * 1e4);
  frame.pitch = (int16_t)(command.pitch * 1e4);
  frame.horizon_distance = (int16_t)(command.horizon_distance * 1e4);
  frame.crc = tools::get_crc16((uint8_t*)&frame.FrameHeader, sizeof(Command_Send_Frame) - 3);
  frame.FrameEnd = 0xAA;
  
  try {
    serial_.write((uint8_t*)&frame, sizeof(Command_Send_Frame));
  } catch (const std::exception & e) {
    tools::logger()->warn("[DM_IMU] send failed: {}", e.what());
  }
}

}  // namespace io
