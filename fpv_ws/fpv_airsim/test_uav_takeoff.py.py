import airsim

# 连接到 AirSim 仿真器
client = airsim.MultirotorClient()
client.confirmConnection()

# 获取控制权并解锁电机
client.enableApiControl(True)
client.armDisarm(True)

# 执行起飞指令 (异步执行，join() 表示等待动作完成)
client.takeoffAsync().join()

# 悬停后降落并上锁
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)