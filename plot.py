import pickle
import matplotlib.pyplot as plt
start_name=['neuron','ppo']
time_path=f'%s_end_time'
score_path=f'%s_score_list.pkl'
x_len=333
for name in start_name:
    end_time=pickle.load(open(time_path%name,'rb'))
    print(name,end_time)
    score_list=pickle.load(open(score_path%name,'rb'))
    save_ratio=x_len/end_time
    dots_num=int(save_ratio*len(score_list))
    x=[i/dots_num*x_len for i in range(dots_num)]
    y=score_list[:dots_num]
    plt.plot(x,y,label=name)
plt.legend(loc='upper left')
plt.xlabel('time(second)')
plt.ylabel('score')
plt.savefig('reward0_3')
plt.show()
