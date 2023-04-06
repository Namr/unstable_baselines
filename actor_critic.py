import gym
import torch
import torch.nn as nn
import numpy as np
import collections
from typing import List

from experience import RolloutBuffer


class ImageActorCriticModel(nn.Module):
    def __init__(self):
        super(ImageActorCriticModel, self).__init__()
               
        # core CNN layers
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())
        
        self.conv_size = self._get_conv_out([1, 84, 84])

        # policy layers
        self.pfc1 = nn.Linear(self.conv_size, 512)
        self.pfc2 = nn.Linear(512, 6)
        
        # value layers
        self.vfc1 = nn.Linear(self.conv_size, 512)
        self.vfc2 = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # apply CNN to image and flatten
        core = self.conv(x)
        core = core.view(x.size()[0], -1)

        # policy head
        policy = nn.functional.relu(self.pfc1(core))
        policy = self.pfc2(policy)
        probs = torch.distributions.Categorical(logits=policy)
        action = probs.sample()

        # value head
        value = nn.functional.relu(self.vfc1(core))
        value = self.vfc2(value)

        return action, probs.log_prob(action), probs.entropy(), value

    def evaluate_value(self, x, actions):
        # apply CNN to image and flatten
        core = self.conv(x)
        core = core.view(x.size()[0], -1)

        # policy head
        policy = nn.functional.relu(self.pfc1(core))
        policy = self.pfc2(policy)
        probs = torch.distributions.Categorical(logits=policy)

        # value head
        value = nn.functional.relu(self.vfc1(core))
        value = self.vfc2(value)

        return probs.log_prob(actions).diag(), probs.entropy(), value


class CartpoleActorCriticModel(nn.Module):
    def __init__(self):
        super(CartpoleActorCriticModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(4, 128)
        self.pfc2 = nn.Linear(128, 2)
        self.vfc2 = nn.Linear(128, 1)

    def forward(self, x):
        core = nn.functional.relu(self.fc1(x))

        policy = self.pfc2(core)
        probs = torch.distributions.Categorical(logits=policy)
        action = probs.sample()

        value = self.vfc2(core)

        return action, probs.log_prob(action), probs.entropy(), value

    def evaluate_value(self, x, actions):
        # flatten and do fully connected pass
        core = nn.functional.relu(self.fc1(x))

        policy = self.pfc2(core)
        probs = torch.distributions.Categorical(logits=policy)

        value = self.vfc2(core)

        return probs.log_prob(actions).diag(), probs.entropy(), value


class A2CAgent():
    def __init__(self, envs: List[gym.Env], model: nn.Module, gamma=0.99, batch_size=128,
                 value_beta=0.5, entropy_beta=0.02, grad_limit=0.5, writer=None):
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for A2CAgent Training")
        
        self.env = env
        self.model = model.to(self.device)
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.value_beta = value_beta
        self.batch_size = batch_size
        self.grad_limit = grad_limit

        self.value = 0.0
        self.final_values = collections.deque(maxlen=100)
        self.reset()
        self.final_values.clear()

        self.memory = RolloutBuffer(self.batch_size, env.observation_space, env.action_space, 1, self.gamma)
        self.last_episode_starts = np.ones((1), dtype=bool)
    
    def reset(self):
        self.state, _ = self.env.reset()
        self.final_values.append(self.value)
        self.value = 0.0
    
    @torch.no_grad()
    def step(self):
        # run observation through network
        obs_tensor = torch.from_numpy(np.array([self.state])).to(self.device)
        action, log_prob, entropy, predicted_value = self.model(obs_tensor)

        # take action according to policy network
        next_observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy().item())
        is_done = terminated or truncated
        
        # remember what happened for training
        self.value += reward
        self.memory.append(self.state, action.cpu(), reward, self.last_episode_starts, predicted_value, log_prob)

        self.state = next_observation
        self.last_episode_starts = is_done

        if is_done:
            self.reset()

        return predicted_value
    
    def train(self, learning_rate, target_reward):
        best_mean_value = -10000000000
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        n = 1
        while True:
            # collect batch and compute returns
            terminal_value = 0.0
            for i in range(self.batch_size):
                terminal_value = self.step()
            self.memory.compute_returns_and_advantage(terminal_value, self.last_episode_starts)
                
            # since batch size is not specified, this loop only happens once
            for old_observations, old_actions, old_values, old_log_probs, advantages, returns in self.memory.get():
                # prep data
                old_actions = torch.Tensor(old_actions).to(self.device).reshape(-1, 1)
                advantages = (advantages - advantages.mean() / (advantages.std() + 1e-8))
                advantages = torch.from_numpy(advantages).to(self.device)
                returns = torch.from_numpy(returns).to(self.device)
                
                # make predicitions
                obs_tensor = torch.from_numpy(old_observations.reshape(-1, *self.env.observation_space.shape)).to(self.device)
                log_probs, entropy, predicted_values = self.model.evaluate_value(obs_tensor, old_actions)
                predicted_values = predicted_values.flatten()
                
                # compute loss
                policy_loss = -(log_probs * advantages).mean()
                value_loss = torch.nn.functional.mse_loss(returns, predicted_values)
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.value_beta * value_loss + self.entropy_beta * entropy_loss

                # backprop and gradient decent
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_limit)
                optimizer.step()
                optimizer.zero_grad()

            self.memory.clear()

            # logging
            mean_value = sum(self.final_values) / len(self.final_values) if len(self.final_values) != 0 else 0
            if mean_value != 0 and n % 4 == 0:
                print(f"Step: {n * self.batch_size} | Loss: {loss} | mean_value: {mean_value}")
                if self.writer is not None:
                    self.writer.add_scalar(f'{self.env.unwrapped.spec.id} A2C Mean Reward', mean_value, n * self.batch_size)

                if mean_value >= best_mean_value:
                    best_mean_value = mean_value
                    torch.save(self.model.state_dict(), f"./models/A2C_{self.env.unwrapped.spec.id}_best.pth")

                    if best_mean_value >= target_reward:
                        print("Solved!")
                        return
            n += 1


class PPOAgent():
    def __init__(self, env: gym.Env, model: nn.Module,
                 gamma=0.99, td_lambda=0.95, trajectory_length=5000, batch_size=256,
                 value_beta=0.5, entropy_beta=0.02, grad_limit=0.5, epsilon=0.1, writer=None):
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for PPOAgent Training")
        
        self.env = env
        self.model = model.to(self.device)
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.entropy_beta = entropy_beta
        self.value_beta = value_beta
        
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        self.grad_limit = grad_limit
        self.epsilon = epsilon

        self.value = 0.0
        self.final_values = collections.deque(maxlen=100)
        self.reset()
        self.final_values.clear()

        self.memory = RolloutBuffer(self.trajectory_length, env.observation_space, env.action_space, 1, self.gamma, self.td_lambda)
        self.last_episode_starts = np.ones((1), dtype=bool)
    
    def reset(self):
        self.state, _ = self.env.reset()
        self.final_values.append(self.value)
        self.value = 0.0
    
    @torch.no_grad()
    def step(self):
        # run observation through network
        obs_tensor = torch.from_numpy(np.array([self.state])).to(self.device)
        action, log_prob, entropy, predicted_value = self.model(obs_tensor)

        # take action according to policy network
        next_observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy().item())
        is_done = terminated or truncated
        
        # remember what happened for training
        self.value += reward
        self.memory.append(self.state, action.cpu(), reward, self.last_episode_starts, predicted_value, log_prob)

        self.state = next_observation
        self.last_episode_starts = is_done

        if is_done:
            self.reset()

        return predicted_value
    
    def train(self, learning_rate, target_reward):
        best_mean_value = -10000000000
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        n = 1
        while True:
            # collect batch and compute returns
            terminal_value = 0.0
            for i in range(self.trajectory_length):
                terminal_value = self.step()
            self.memory.compute_returns_and_advantage(terminal_value, self.last_episode_starts)
                
            # since batch size is not specified, this loop only happens once
            for old_observations, old_actions, old_values, old_log_probs, advantages, returns in self.memory.get(self.batch_size):
                # prep data
                old_actions = torch.Tensor(old_actions).to(self.device).reshape(-1, 1)
                advantages = (advantages - advantages.mean() / (advantages.std() + 1e-8))
                advantages = torch.from_numpy(advantages).to(self.device)
                returns = torch.from_numpy(returns).to(self.device)
                old_log_probs = torch.from_numpy(old_log_probs).to(self.device)
                
                # make predicitions
                obs_tensor = torch.from_numpy(old_observations.reshape(-1, *self.env.observation_space.shape)).to(self.device)
                log_probs, entropy, predicted_values = self.model.evaluate_value(obs_tensor, old_actions)
                predicted_values = predicted_values.flatten()
                
                # compute loss

                # clip policy ratio loss
                ratio = torch.exp(log_probs - old_log_probs)
                unclipped_policy_loss = advantages * ratio
                clipped_policy_loss = advantages * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                policy_loss = -torch.min(unclipped_policy_loss, clipped_policy_loss).mean()
                
                # TODO: implement value clipping
                value_loss = torch.nn.functional.mse_loss(returns, predicted_values)
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.value_beta * value_loss + self.entropy_beta * entropy_loss

                # backprop and gradient decent
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_limit)
                optimizer.step()
                optimizer.zero_grad()

            self.memory.clear()

            # logging
            mean_value = sum(self.final_values) / len(self.final_values) if len(self.final_values) != 0 else 0
            if mean_value != 0:
                print(f"Step: {n * self.trajectory_length} | Loss: {loss} | mean_value: {mean_value}")
                if self.writer is not None:
                    self.writer.add_scalar(f'{self.env.unwrapped.spec.id} PPO Mean Reward', mean_value, n * self.trajectory_length)

                if mean_value >= best_mean_value:
                    best_mean_value = mean_value
                    torch.save(self.model.state_dict(), f"./models/PPO_{self.env.unwrapped.spec.id}_best.pth")

                    if best_mean_value >= target_reward:
                        print("Solved!")
                        return
            n += 1

