# 虚幻引擎 与 AirSim 构建
**(Ubuntu 22.04 && UE 4.27)**
## 克隆并编译 UE 4.27
```bash
  git clone -b 4.27 https://github.com/EpicGames/UnrealEngine.git
  cd UnrealEngine
  ./Setup.sh
  ./GenerateProjectFiles.sh
  make
```
### 编译成功后打开 UE
```bash
  cd UnrealEngine
  ./Engine/Binaries/Linux/UE4Editor
```

## 克隆并编译 AirSim
[build_linux.md](https://github.com/Microsoft/AirSim/blob/main/docs/build_linux.md)
```bash
  git clone https://github.com/microsoft/AirSim.git
  cd AirSim
  ./setup.sh
  ./build.sh
```
### 解决 ./setup.sh 的可能报错
旧版本的 vulkan-utils 包已经被重命名为了 vulkan-tools
```bash
  sed -i 's/vulkan-utils/vulkan-tools/g' setup.sh
```
系统默认的 Clang 编译器版本较新，脚本中硬编码的 -8 后缀会导致 apt 索引彻底失效
打开并修改setup.sh脚本中的安装命令行为：```sudo apt-get install -y clang libc++-dev libc++abi-dev```

### 解决 ./build.sh 的可能报错
Clang 版本的可能报错 以及 其他可能配置的硬编码报错
```bash
  rm -rf build_release
  sed -i 's/clang-8/clang/g' build.sh
  sed -i 's/clang++-8/clang++/g' build.sh
  grep -rl "c++fs" . | xargs -r sed -i 's/-lc++fs//g'
  grep -rl "c++fs" . | xargs -r sed -i 's/c++fs//g'
```

## Blocks 基础项目
[配置Blocks环境](https://frendowu.github.io/AirSim-docs-zh/unreal_blocks/)
### 使用终端手动编译项目
直接启动 UE 仿真会提示 Blocks 模块缺失或需要重新编译 Please build through your IDE.
```bash
  cd AirSim/Unreal/Environments/Blocks
  #强制删除编译缓存文件夹
  rm -rf Binaries Intermediate Saved
  #运行更新脚本
  ./update_from_git.sh
  #手动调用 Unreal Build Tool (UBT) 编译 ( ~/UnrealEngine 和 ~/AirSim 替换为实际的安装路径)
  ~/UnrealEngine/Engine/Build/BatchFiles/Linux/Build.sh BlocksEditor Linux Development ~/AirSim/Unreal/Environments/Blocks/Blocks.uproject
```
### 启动Blocks仿真
  ```bash
  # ~/UnrealEngine 和 ~/AirSim 替换为实际路径
  ~/UnrealEngine/Engine/Binaries/Linux/UE4Editor ~/AirSim/Unreal/Environments/Blocks/Blocks.uproject
  ```
  - 进入编辑器画面后，点击正上方的 Play (播放) 按钮
  - 如果屏幕弹出一个对话框问你选择哪种载具，请选择 Multirotor (多旋翼无人机)
  - 如果无人机成功出现在场景中，恭喜！
  - 简单无人机起飞测试。保持虚幻引擎在 Play 状态，运行 Python 基础起飞脚本 test_uav_takeoff.py (安装 Python 客户端包 ```pip3 install msgpack-rpc-python airsim```)
  - 如果无人机能够顺利起飞并降落，恭喜！
  - （注：关闭 UE4 的后台休眠设置。进入“Edit->Editor Preferences“并搜索”CPU“,取消勾选 Use Less CPU when in Background ）

# ROS2 Wrapper
## 配置 AirSim
修改文件 ~/Documents/AirSim/settings.json （绝对路径）
正确配置后在 UE 编辑器画面点击 Play (播放) 按钮后不再出现弹窗，会直接生成无人机
```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "PX4": {
      "VehicleType": "PX4Multirotor",
      "UseSerial": false,
      "LockStep": true,
      "UseTcp": true,
      "TcpPort": 4560,
      "ControlIp": "127.0.0.1",
      "ControlPort": 14580,
      "Cameras": {
        "front_cam": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.1, "Y": 0, "Z": 0,
          "Pitch": 0, "Roll": 0, "Yaw": 0
        }
      }
    }
  }
}
```
注：这里配置了一个名为 front_cam 的相机，前向延伸 0.1 米，输出 640x480 的 RGB 图像（ImageType: 0）。可根据需要进行调整。
## 编译 ROS2 
### 物理隔离 ROS2 Wrapper 
另外克隆一个AirSim并更名为AirSim_ROS
  ```bash
    mv AirSim AirSim_ROS
    cd AirSim_ROS
    ./setup.sh
  ```
抹除 AirSim 核心的 LLVM 强制 libc++ 设定,使用和 ROS2 一样的 libstdc++ 进行编译！
  ```bash
    grep -rl "stdlib=libc++" . | xargs -r sed -i 's/-stdlib=libc++//g'
    grep -rl "c++abi" . | xargs -r sed -i 's/-lc++abi//g'
  ```
重新编译AirSim_ROS:
  ```bash
    rm -rf build_release
    ./build.sh
  ```

### 修改CMakeLists.txt
```cd AirSim_ROS/ros2```
打开 airsim_ros_pkgs 功能包的 CMakeLists.txt 配置文件在其最后添加依赖包的链接绑定：
```c++
ament_target_dependencies(airsim_ros tf2 tf2_ros tf2_geometry_msgs tf2_sensor_msgs mavros_msgs std_srvs)
ament_target_dependencies(airsim_node tf2 tf2_ros tf2_geometry_msgs tf2_sensor_msgs mavros_msgs std_srvs)
ament_target_dependencies(pd_position_controller_simple tf2 tf2_ros tf2_geometry_msgs tf2_sensor_msgs mavros_msgs std_srvs)
```
### 切除 Wrapper 的强行解锁
```cd AirSim_ROS/ros2/src/airsim_ros_pkgs/src```
- 打开 airsim_ros_pkgs 功能包的 airsim_ros_wrapper.cpp 源码
- 找到并注释强行解锁指令：
  ```c++
    // airsim_client_->armDisarm(true, vehicle_name);
    // airsim_client_->enableApiControl(true, vehicle_name);
  ```
### 编译ROS2 && 启动 Wrapper
```bash
  cd ~/AirSim_ROS/ros2
  source /opt/ros/humble/setup.bash
  rm -rf build/ install/ log/
  #使用 Clang 编译
  export CC=clang && export CXX=clang++
  colcon build
```
```bash
  source install/setup.bash
  ros2 launch airsim_ros_pkgs airsim_node.launch.py
```
可在终端查看有无 ros2 话题 ``` ros2 topic list``` 和其频率 ```ros2 topic hz 话题名称```

## 架构运行顺序
1. **启动 PX4 SITL：** 在 PX4 源码目录下运行 ```make px4_sitl none_iris```（这会监听 4560 端口等待 AirSim 接入）
2. **启动 AirSim：** 在 UE 中点击 Play 运行 Blocks 环境 (等待PX4终端提提示：Ready for takeoff! ) 
3. **启动 AirSim ROS2 Wrapper：** 建立 ROS2 与 AirSim 的图像/状态桥梁 (可用 rqt 工具查看相机画面)


