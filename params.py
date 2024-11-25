#expironment
epochs=100  #筛选轮次
longest_time=1e4    #模拟筛选的时间
dt=0.1  # 神经元更新的间隔

#enviroment
length=7  # 环境长度
sight=3  # 视野范围
num_food=1  # 食物数量
#neuron
### 常量
C_m = 1.0  # 膜电容 (µF/cm²)
g_L = 0.1  # 漏导 (mS/cm²)
E_L = -70.0  # 漏电反转电位 (mV)
fire_threshold = -50.0  # 火灾阈值 (mV)

