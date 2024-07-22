from copy import deepcopy
from .models import *
import time
import torch

class Agent(object):

    def __init__(self, envs,args, device, discount=0.99, tau=0.005):
        self.device = device
        self.args = args
        self.discount = discount
        self.tau = tau
        self.actor = Actor(envs).to(device)
        self.global_step = 0
        self.start_time = time.time()
        self.next_obs = torch.Tensor(envs.reset()).to(device)
        self.next_done = torch.zeros(args.num_envs).to(device)
        self.batch_size = int(args.num_envs * args.num_steps)
        self.num_updates = args.total_timesteps // args.batch_size

        
        # self.actor_target = deepcopy(self.actor)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(envs).to(device)
        # self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.critic_optimizer, start_factor=1, end_factor=0, total_iters=self.num_updates)

        
    # def select_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     return self.actor(state).cpu().data.numpy().flatten()
    
    # @staticmethod
    # def soft_update(local_model, target_model, tau):
    #     for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #         target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
    # def save_checkpoint(self, filename):
    #     torch.save(self.critic.state_dict(), filename + '_critic')
    #     torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')
    #     torch.save(self.actor.state_dict(), filename + '_actor')
    #     torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')
        
    # def load_checkpoint(self, filename):
    #     self.critic.load_state_dict(
    #         torch.load(
    #             filename + "_critic",
    #             map_location=torch.device('cpu')
    #         )
    #     )
    #     self.critic_optimizer.load_state_dict(
    #         torch.load(
    #             filename + "_critic_optimizer",
    #             map_location=torch.device('cpu')
    #         )
    #     )
    #     self.critic_target = deepcopy(self.critic)
    #     self.actor.load_state_dict(
    #         torch.load(
    #             filename + "_actor",
    #             map_location=torch.device('cpu')
    #         )
    #     )
    #     self.actor_optimizer.load_state_dict(
    #         torch.load(
    #             filename + "_actor_optimizer",
    #             map_location=torch.device('cpu')
    #         )
    #     )
    #     self.actor_target = deepcopy(self.actor)
        
        
    def train(self, memory):
       for update in range(1, self.num_updates + 1):
        if self.args.anneal_lr:
            self.scheduler.step()

        
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        target_q = self.critic_target(next_state, self.actor_target(next_state))
        target_q = reward + ((1-done) * self.discount * target_q).detach()
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        DDPGAgent.soft_update(self.critic, self.critic_target, self.tau)
        DDPGAgent.soft_update(self.actor, self.actor_target, self.tau)