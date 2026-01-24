#ifndef IO__Dm_Imu_HPP
#define IO__Dm_Imu_HPP

#include <math.h>
#include <serial/serial.h>
#include <Eigen/Geometry>
#include <array>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <thread>

#include "tools/thread_safe_queue.hpp"
#include "io/command.hpp"

namespace io
{

enum Mode
{
  idle,
  auto_aim,
  small_buff,
  big_buff,
  outpost
};
const std::vector<std::string> MODES = {"idle", "auto_aim", "small_buff", "big_buff", "outpost"};

enum ShootMode
{
  left_shoot,
  right_shoot,
  both_shoot
};
const std::vector<std::string> SHOOT_MODES = {"left_shoot", "right_shoot", "both_shoot"};

struct __attribute__((packed)) IMU_Receive_Frame
{
  uint8_t FrameHeader1;
  uint8_t flag1;
  uint8_t slave_id1;
  uint8_t reg_acc;
  uint32_t accx_u32;
  uint32_t accy_u32;
  uint32_t accz_u32;
  uint16_t crc1;
  uint8_t FrameEnd1;

  uint8_t FrameHeader2;
  uint8_t flag2;
  uint8_t slave_id2;
  uint8_t reg_gyro;
  uint32_t gyrox_u32;
  uint32_t gyroy_u32;
  uint32_t gyroz_u32;
  uint16_t crc2;
  uint8_t FrameEnd2;

  uint8_t FrameHeader3;
  uint8_t flag3;
  uint8_t slave_id3;
  uint8_t reg_euler;
  uint32_t roll_u32;
  uint32_t pitch_u32;
  uint32_t yaw_u32;
  uint16_t crc3;
  uint8_t FrameEnd3;
};

struct __attribute__((packed)) Control_Receive_Frame
{
  uint8_t FrameHeader;
  uint8_t flag;
  uint8_t slave_id;
  uint8_t reg_control;
  uint16_t bullet_speed_u16;
  uint8_t mode;
  uint8_t shoot_mode;
  uint16_t ft_angle_u16;
  uint16_t crc;
  uint8_t FrameEnd;
};

struct __attribute__((packed)) Command_Send_Frame
{
  uint8_t FrameHeader;
  uint8_t flag;
  uint8_t slave_id;
  uint8_t reg_command;
  uint8_t control;
  uint8_t shoot;
  int16_t yaw;
  int16_t pitch;
  int16_t horizon_distance;
  uint16_t crc;
  uint8_t FrameEnd;
};

typedef struct
{
  float accx;
  float accy;
  float accz;
  float gyrox;
  float gyroy;
  float gyroz;
  float roll;
  float pitch;
  float yaw;
} IMU_Data;

class DM_IMU
{
public:
  double bullet_speed;
  Mode mode;
  ShootMode shoot_mode;
  double ft_angle;

  DM_IMU();
  ~DM_IMU();

  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);
  
  void send(Command command);

private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  void init_serial();
  void get_imu_data_thread();

  serial::Serial serial_;
  std::thread rec_thread_;

  tools::ThreadSafeQueue<IMUData> queue_;
  IMUData data_ahead_, data_behind_;

  std::atomic<bool> stop_thread_{false};
  IMU_Receive_Frame receive_data{};
  Control_Receive_Frame control_data{};
  IMU_Data data{};
};

}  // namespace io

#endif



// #ifndef IO__Dm_Imu_HPP
// #define IO__Dm_Imu_HPP

// #include <math.h>
// #include <serial/serial.h>

// #include <Eigen/Geometry>
// #include <array>
// #include <fstream>
// #include <initializer_list>
// #include <iostream>
// #include <thread>

// #include "tools/thread_safe_queue.hpp"

// namespace io
// {

// struct __attribute__((packed)) IMU_Receive_Frame
// {
//   uint8_t FrameHeader1;
//   uint8_t flag1;
//   uint8_t slave_id1;
//   uint8_t reg_acc;
//   uint32_t accx_u32;
//   uint32_t accy_u32;
//   uint32_t accz_u32;
//   uint16_t crc1;
//   uint8_t FrameEnd1;

//   uint8_t FrameHeader2;
//   uint8_t flag2;
//   uint8_t slave_id2;
//   uint8_t reg_gyro;
//   uint32_t gyrox_u32;
//   uint32_t gyroy_u32;
//   uint32_t gyroz_u32;
//   uint16_t crc2;
//   uint8_t FrameEnd2;

//   uint8_t FrameHeader3;
//   uint8_t flag3;
//   uint8_t slave_id3;
//   uint8_t reg_euler;  //r-p-y
//   uint32_t roll_u32;
//   uint32_t pitch_u32;
//   uint32_t yaw_u32;
//   uint16_t crc3;
//   uint8_t FrameEnd3;
// };

// typedef struct
// {
//   float accx;
//   float accy;
//   float accz;
//   float gyrox;
//   float gyroy;
//   float gyroz;
//   float roll;
//   float pitch;
//   float yaw;
// } IMU_Data;

// class DM_IMU
// {
// public:
//   DM_IMU();
//   ~DM_IMU();

//   Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);

// private:
//   struct IMUData
//   {
//     Eigen::Quaterniond q;
//     std::chrono::steady_clock::time_point timestamp;
//   };

//   void init_serial();
//   void get_imu_data_thread();

//   serial::Serial serial_;
//   std::thread rec_thread_;

//   tools::ThreadSafeQueue<IMUData> queue_;
//   IMUData data_ahead_, data_behind_;

//   std::atomic<bool> stop_thread_{false};
//   IMU_Receive_Frame receive_data{};  //receive data frame
//   IMU_Data data{};
// };

// }  // namespace io

// #endif
