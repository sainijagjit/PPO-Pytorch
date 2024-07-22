import torch
import torch.nn.functional as F
from utils import *


class Actor(torch.nn.Module):
    
    def __init__(self,envs):
        super(Actor, self).__init__()
        self.l1 = layer_init(torch.nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.l2 = layer_init(torch.nn.Linear(64, 64))
        self.l3 = layer_init(torch.nn.Linear(64, envs.single_action_space.n),std=0.01)
    
    def forward(self,state):
        a = F.tanh(self.l1(state))
        a = F.tanh(self.l2(state))
        a = self.l3(a)
        return a
        
        
class Critic(torch.nn.Module):

    def __init__(self,envs):
        super(Critic, self).__init__()
        self.l1 = layer_init(torch.nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64))
        self.l2 = layer_init(torch.nn.Linear(64, 64))
        self.l3 = layer_init(torch.nn.Linear(64, envs.single_action_space.n),std=1)

    def forward(self,state):
        a = F.tanh(self.l1(state))
        a = F.tanh(self.l2(state))
        a = self.l3(a)
        return a