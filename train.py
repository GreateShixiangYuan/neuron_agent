import pickle
import random
import time
from multiprocessing import Pool, cpu_count
from network import Agent
from environment import Environment
from params import *
import numpy as np
import copy
longest_time=6000
layers=[6,6]
attribute={'food':1,'boundary':0,'move':-0.05,'static':-0.05}#food boundry move static
# 评估当前参数的质量
def evaluate_agent(params,turns=1):
    agent = Agent(layers,attribute)#由于采用多进程，每个agent和环境有自己的状态参数，故需重新创建
    env = Environment()
    score_list=[]
    for i in range(turns):
        state = env.reset()
        agent.init(params)
        score = 0
        while True:
            action = agent.update(state)
            state = env.step(action)
            score += 1
            if not agent.if_alive() or score>longest_time:
                break
        score_list.append(score)
    return sum(score_list)/turns


if __name__ == "__main__":
    # 初始化参数
    param_size = 5000 
    final_mu = np.zeros(param_size)
    neuron_score_list=np.zeros(10)
    neuron_end_time=0
    for k in range(1):
        start_time=time.time()
        agent = Agent(layers)
        agent_num = 1000#根据均值和方差生成多少个agent参数
        select_rate = 0.01#筛选最好的前百分之几
        
          #最大参数尺寸，比agent的参数量大就行
        mu = np.random.uniform(low=-1, high=1, size=param_size)
        mu[:agent.param_len] = agent.get_parameters()#用agent默认初始化参数初始化mu,mu是每个参数对应的均值
        sigma = np.zeros(param_size) #生成每个参数的方差
        sigma[:agent.param_len]=np.abs(mu[:agent.param_len])#用mu大小初始化方差，有一定合理性
        # mu=pickle.load(open("mu1.pkl", "rb"))
        # sigma=pickle.load(open("sigma1.pkl", "rb"))
        sigma_scale=1
        min_sigma = 0.0001#最小方差
        noise_ratio=1
        epochs = 200
        # 多进程池
        pool = Pool(processes=cpu_count())  # 使用 CPU 核心数创建进程池
        record_list=[]

        for epoch in range(epochs):
            # 生成随机参数表
            param_table = np.tile(mu, (agent_num, 1))
            random_matrix = np.random.rand(agent_num, param_size)
            noise_mask = (random_matrix < noise_ratio).astype(int)
            noise = np.random.normal(0, sigma, size=(agent_num, param_size))
            param_table += noise_mask * noise


            # 并行计算每个代理的得分
            score_list = pool.map(evaluate_agent, param_table)

            # 选择顶尖代理的参数
            rank_list = np.argsort(score_list)[::-1]
            select_index = rank_list[:int(agent_num * select_rate)]
            select_param = param_table[select_index]#选择表现最好的那一部分参数

            # 用其更新均值和标准差
            mu = np.mean(select_param, axis=0)
            agent.init(mu)#用agent裁剪机制防止均值偏离太大，训练更稳定
            mu[:agent.param_len] = agent.get_parameters()
            sigma = np.std(select_param, axis=0)*sigma_scale + min_sigma#标准差加上最小标准差，保证一定探索

            # 打印进度信息
            print("Epoch:", epoch, "Score:", np.mean(score_list),"select:",np.mean([score_list[i] for i in select_index]))
            record_list.append(np.mean([score_list[i] for i in select_index]).item())
            # 保存结果
            pickle.dump(mu, open("mu1.pkl", "wb"))
            pickle.dump(sigma, open("sigma1.pkl", "wb"))
            pickle.dump(np.mean(score_list), open("score1.pkl", "wb"))
        final_mu+=mu
        neuron_score_list+=np.array(record_list)
        neuron_end_time+=time.time()-start_time
        pool.close()
        pool.join()

    pickle.dump(final_mu, open("final_mu2.pkl", "wb"))
    print(final_mu[agent.param_len-2])
    neuron_end_time/=5
    neuron_score_list/=5
    pickle.dump(neuron_score_list,open("neuron_score_list.pkl","wb"))
    pickle.dump(neuron_end_time,open("neuron_end_time","wb"))

