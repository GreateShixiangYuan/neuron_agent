import random
from params import *
class Environment:
    def __init__(self):
        self.half_length=length//2
        self.half_sight=sight//2
        self.reset()
    def reset(self):
        self.position=0
        self.food=[]
        for i in range(num_food):
            self.generate_food()
        return self.get_state()
    def generate_food(self):
        # while True:
        #     x=random.randint(-self.half_length,self.half_length)
        #     if x not in self.food and x!=self.position:
        #         self.food.append(x)
        #         break
        if self.position==-2:
            self.food=[2]
        else:
            self.food=[-2]
    def get_state(self):
        state=[]
        #0 safe,-1 danger, 1 food
        for i in range(self.position-self.half_sight,self.position+self.half_sight+1):
            if i<-self.half_length or i>self.half_length:
                state.append(-1)
            elif i in self.food:
                state.append(1)
            else:
                state.append(0)
        return state
    def step(self,action):
        if self.position in self.food:
            self.food.remove(self.position)
            self.generate_food()
        temp=self.position
        if action==1:
            self.position-=1
        elif action==2:
            self.position+=1
        else:
            assert action==0
        if abs(self.position)>self.half_length+1:   #不是不然出边界，可以出边界，这样才有惩罚，才能观测到当前处于边界外
            self.position=temp
        return self.get_state()
