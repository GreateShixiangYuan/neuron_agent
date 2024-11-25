import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import Environment
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2=nn.Linear(128,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class Actor(nn.Module):
    def __init__(self,num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2=nn.Linear(128,num_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x
class Agent:
    def __init__(self,actor_lr,critic_lr,gamma,device):
        self.actor=Actor(3).to(device)
        self.critic=Critic().to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma=gamma
        self.device=device
    def get_action(self,observation):
        observation=torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        action_probs=self.actor(observation)
        action=torch.argmax(action_probs,dim=1).item()
        return action
    def update(self,observation,action,reward,next_observation,done):
        observation=torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        next_observation=torch.FloatTensor(next_observation).unsqueeze(0).to(self.device)
        action=torch.LongTensor([action]).unsqueeze(0).to(self.device)
        reward=torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        done=torch.FloatTensor([done]).unsqueeze(0).to(self.device)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        action_probs=self.actor(observation)
        critic_value=self.critic(observation)
        next_critic_value=self.critic(next_observation)
        target_value=reward+self.gamma*next_critic_value*(1-done)
        advantage=target_value-critic_value
        actor_loss=-torch.log(action_probs[0][action])*advantage.detach()
        actor_loss=actor_loss.mean()
        critic_loss=F.mse_loss(target_value,critic_value)
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
device='cuda:0'
actor_lr=0.001
critic_lr=0.001
gamma=0.8
agent=Agent(actor_lr,critic_lr,gamma,device)
# agent.actor.load_state_dict(torch.load("actor.pkl"))
# agent.critic.load_state_dict(torch.load("critic.pkl"))
env=Environment()
epochs=20000
longest_time=6000
for epoch in range(epochs):
    state=env.reset()
    score=0
    energy=10
    done=False
    while not done:
        action=agent.get_action(state)
        next_state=env.step(action)
        score+=1
        if energy<=0 or score>longest_time:
            done=True
        reward=0
        if next_state[1]==1:
            reward=1
        elif next_state[1]==-1:
            reward=-1
        else:
            reward=-0.05
        energy+=reward
        agent.update(state,action,reward,next_state,done)
        state=next_state
    if epoch%1==0:
        print("Epoch:",epoch,"Score:",score)
torch.save(agent.actor.state_dict(),"actor.pkl")
torch.save(agent.critic.state_dict(),"critic.pkl")