import numpy as np
from params import *


class Agent:
    def __init__(self,layers=[6,6,2],attribute=None,init_parameters=None):
        self.layers = layers    #网络结构，每层多少个
        self.attribute=attribute#食物 边界 移动 静止
        assert len(layers) >= 2 #至少有sensor和一层神经元
        assert layers[0]%3 == 0 #观察长度是3，最好3的倍数
        assert layers[-1]%2 == 0#最后一层输入ddm，一向输出向左一半向右
        self.sensors = [Sensor_Neuron() for i in range(layers[0])]#sensors根据环境观察发放特点频率刺激
        self.neurons=[]#计算复杂逻辑函数
        #layers比neurons多一层sensor
        for i in range(1,len(layers)):
            self.neurons.append([Neuron(layers[i-1]) for j in range(layers[i])])
        self.ddm = DDM()
        self.param_len=6*layers[0]  #所有参数从一维参数上取，方便管理，这里计算该agent一共多少参数
        for j in range(len(self.neurons)):
            param_size = 3 + self.layers[j] * 2
            for neuron in self.neurons[j]:
                self.param_len += param_size
        self.param_len+=2
        if init_parameters is not None:
            self.init(init_parameters)
    def init(self,init_parameters):#参数由外部管理和传入
        self.energy = 10
        self.alive = True
        i = 0
        sensor_size = 6 #一个sensor的参数量
        for sensor in self.sensors:
            sensor.init(init_parameters[i:i + sensor_size])
            i += sensor_size
        for j in range(len(self.neurons)):
            param_size = 3 + self.layers[j] * 2 #一个神经元的参数
            for neuron in self.neurons[j]:
                neuron.init(init_parameters[i:i + param_size])
                i += param_size
        ddm_size = 2    #ddm参数
        self.ddm.init(init_parameters[i:i + ddm_size])
        i += ddm_size
        assert self.param_len==i
        self.clip()
    def get_parameters(self):#返回所有参数
        parameters = []
        for sensor in self.sensors:
            parameters.append(sensor.get_parameters())
        for i in range(len(self.neurons)):
            for neuron in self.neurons[i]:
                parameters.append(neuron.get_parameters())
        parameters.append(self.ddm.get_parameters())
        parameters=np.concatenate(parameters)
        assert len(parameters) == self.param_len
        return parameters
    def clip(self): #裁剪所有参数，仍训练更稳定
        for sensor in self.sensors:
            sensor.clip()
        for i in range(len(self.neurons)):
            for neuron in self.neurons[i]:
                neuron.clip()
        self.ddm.clip()
    def update(self, observation):
        now_id=len(observation)//2#当前位置下标
        if observation[now_id] == -1:#当前是边界
            self.energy += self.attribute['boundary']
        elif observation[now_id] == 1:
            self.energy += self.attribute['food']
        if self.energy < 0:
            self.alive = False
            return 0
        #传递更新状态
        activation_signals = [self.sensors[i].update(observation[i % 3]) for i in range(len(self.sensors))]
        for i in range(len(self.neurons)):
            activation_signals = [neuron.update(activation_signals) for neuron in self.neurons[i]]
        #从ddm获取动作
        action = self.ddm.update(activation_signals)     
        if action==0:   #根据有没有运动消耗能量
            self.energy+=self.attribute['static']
        else:
            self.energy+=self.attribute['move'] 
        return action

    def if_alive(self):
        return self.alive


class Neuron:
    def __init__(self, num_parent,init_parameters=None):
        self.num_parent = num_parent
        self.fire_threshold = fire_threshold
        if init_parameters is not None:
            self.init(init_parameters)
        else:
            self.activation = False
            self.rest_fire_time = 0
            self.rest_refractory_time = 0
            self.V = -70
            self.max_fire_time = 1
            self.max_refractory_time = 1
            self.E_syn = np.random.uniform(-120, 50, num_parent)
            self.g_syn = np.random.uniform(-0.1, 0.3, num_parent)
    def init(self, init_parameters):
        self.activation = False
        self.rest_fire_time = 0
        self.rest_refractory_time = 0
        self.V = init_parameters[0]
        self.max_fire_time = init_parameters[1]
        self.max_refractory_time = init_parameters[2]
        self.E_syn = init_parameters[3:3+self.num_parent]
        self.g_syn = init_parameters[3+self.num_parent:3+self.num_parent*2]
    def get_parameters(self):
        parameters=np.array([self.V,self.max_fire_time,self.max_refractory_time])
        return np.concatenate((parameters,self.E_syn,self.g_syn))
    def clip(self):
        self.V = np.clip(self.V, -1000, 1000)
        self.max_fire_time = np.clip(self.max_fire_time, -2, 10)
        self.max_refractory_time = np.clip(self.max_refractory_time, -2, 10)
        self.E_syn = np.clip(self.E_syn, -120, 50)
        self.g_syn = np.clip(self.g_syn, -0.1, 0.3)

    def update(self, fire_signals):
        fire_signals = np.array(fire_signals, dtype=float)
        # 如果不应期则不更新电压，并减小不应期；否则正常更新
        if self.rest_refractory_time == 0:
            I_L = g_L * (self.V - E_L)  # 漏电流
            I_syn = self.g_syn * (self.V - self.E_syn) * fire_signals  # 突触电流
            self.V += dt * (1 / C_m) * (-I_L - I_syn.sum())
        else:
            self.rest_refractory_time -= dt
            if self.rest_refractory_time <= 0:
                self.rest_refractory_time = 0
        #rest_fire_time模拟神经递质释放时间。如果处于释放过程，神经元持续激活。不在释放，如果当前电位达到阈值，开始释放过程
        if self.rest_fire_time > 0:
            self.rest_fire_time -= dt
        if self.V >= self.fire_threshold:
            self.activation = True
            self.rest_fire_time = self.max_fire_time
            self.rest_refractory_time = self.max_refractory_time
            self.V = -70
        if self.rest_fire_time <= 0:
            self.activation = False
            self.rest_fire_time = 0
        self.V = np.clip(self.V, -1000, 1000)#有且仅有电压参数变化，裁剪之
        return self.activation


class Sensor_Neuron:
    def __init__(self,init_parameters=None):
        if init_parameters is not None:
            self.init(init_parameters)
        else:
            self.activation = False
            self.rest_fire_time = 0
            self.fire_frequency = np.array([0.1, 0.1, 0.1])
            self.max_fire_time = np.array([1, 1, 1])
    def init(self, init_parameters):
        self.activation = False
        self.rest_fire_time = 0     #也模拟神经递质释放时间
        self.fire_frequency = init_parameters[0:3]#对3种情况，有3种激活频率，也有三种释放时间
        self.max_fire_time = init_parameters[3:6]
    def get_parameters(self):
        return np.concatenate((self.fire_frequency, self.max_fire_time))
    def clip(self):
        self.fire_frequency = np.clip(self.fire_frequency, -0.1, 1)
        self.max_fire_time = np.clip(self.max_fire_time, -2, 10)
    def update(self, sensor_signals):
        # sensor_signals: -1 ,0 , 1
        fire_frequency = self.fire_frequency[sensor_signals + 1]#根据输入选择对应激活频率和释放时间
        if not self.activation:  # 没有被激活时有一定的频率激活该神经元
            if np.random.rand() < fire_frequency:
                self.activation = True
                self.rest_fire_time = self.max_fire_time[sensor_signals + 1]
        else:
            self.rest_fire_time -= dt
            if self.rest_fire_time <= 0:
                self.activation = False
                self.rest_fire_time = 0
        return self.activation
class DDM:
    def __init__(self,init_parameters=None):
        # --- DDM Parameters ---
        self.v = 1  #信号积累速度
        if init_parameters is not None:
            self.init(init_parameters)
        else:
            self.stimulus = 0
            self.decision_boundary = 5
            self.sigma = 0.01
    def init(self, init_parameters):
        self.stimulus = 0   #当前信号
        self.decision_boundary = init_parameters[0]#决策阈值
        self.sigma = init_parameters[1] #随机扰动
    def get_parameters(self):
        return np.array([self.decision_boundary, self.sigma])
    def clip(self):
        self.decision_boundary = np.clip(self.decision_boundary, 0, 100)
        self.sigma = np.clip(self.sigma, 0, 1)
    def update(self, activation_signal):
        fire_signals = np.array(activation_signal, dtype=float)#前半向左
        left_signal = np.sum(fire_signals[:len(fire_signals)//2])
        right_signal = np.sum(fire_signals[len(fire_signals)//2:])
        judge_direction = left_signal - right_signal
        action = self.simulate_ddm(judge_direction)#累积刺激
        # if left_signal>right_signal:
        #     action=1
        # elif right_signal>left_signal:
        #     action=2
        # else:
        #     # action=np.random.choice([-1,1])
        #     action=1
        return action

    # Simulation Function
    def simulate_ddm(self, direction):
        self.stimulus += (direction * self.v + self.sigma * np.random.randn())#累积刺激
        action = 0
        if abs(self.stimulus) >= self.decision_boundary:#达到决策阈值，决策；否则输出0静止
            if self.stimulus > 0:
                action = 1
            else:
                action = 2
            self.stimulus = 0   #完成一次决策，刺激积累清零
        # if action==0:
        #     action=0
        return action