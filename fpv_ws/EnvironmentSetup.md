必要准备：
|||
|-|-|
| Ubuntu22.04 | ROS2（humble）|
|||

[Ubuntu22.04镜像下载](https://mirrors.ustc.edu.cn/ubuntu-releases/22.04/)

注：若使用虚拟机软件VMware （自行下载安装）注意设置里关闭3D图形加速，建议使用双系统

  

安装ROS2（humble）:

```bash

wget  http://fishros.com/install  -O  fishros && bash  fishros

```

  

安装PX4: [ROS 2 用户指南 | PX4 Guide (main)](https://docs.px4.io/main/zh/ros2/user_guide)



Ubuntu上配置 PX4 开发环境：

```bash

git  clone  https://github.com/PX4/PX4-Autopilot.git  --recursive

bash  ./PX4-Autopilot/Tools/setup/ubuntu.sh

cd  PX4-Autopilot/

make  px4_sitl

```

配置微型 XRCE-DDS 代理与客户端：

```bash

git  clone  -b  v2.4.3  https://github.com/eProsima/Micro-XRCE-DDS-Agent.git

cd  Micro-XRCE-DDS-Agent

mkdir  build

cd  build

cmake  ..

make

sudo  make  install

sudo  ldconfig  /usr/local/lib/

```

  

#gz garden版本（使用）：

```bash

sudo  wget  https://packages.osrfoundation.org/gazebo.gpg  -O  /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo  "deb [arch=$(dpkg  --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release  -cs) main" | sudo  tee  /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo  apt-get  update

sudo  apt-get  install  gz-garden

sudo  apt-get  install  ros-humble-ros-gzgarden

```

  

#网络ip释放获取：

sudo dhclient -r

Gazebo仿真:

-  启动DDS代理：```MicroXRCEAgent udp4 -p 8888```

- 打开QGC

- 运行PX4

```PX4_GZ_WORLD=default make px4_sitl gz_x500 ```
or ```PX4_GZ_WORLD=default make px4_sitl gz_x500_mono_cam```

#桥接

```bash

ros2  run  ros_gz_bridge  parameter_bridge  /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image

```

#生球

```bash

#'sdf_filename:地址按实际配置

gz  service  --service  /world/default/create  --reqtype  gz.msgs.EntityFactory  --reptype  gz.msgs.Boolean  --timeout  1000  --req  'sdf_filename: "/home/shuai/sh_ws/src/qjwdds/gz/models/sptball.sdf"'

```

rqt

```bash

ros2  run  rqt_image_view  rqt_image_view

```

  

#运动

```bash

gz  topic  -t  /model/sptballoon/joint/y_axis_joint/0/cmd_pos  -m  gz.msgs.Double  -p  'data: 5.0'

```


```bash

chmod  +x  loop_trajectory.sh

./loop_trajectory.sh

```