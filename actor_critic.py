import gym
import torch
import torch.nn as nn
import numpy as np
import collections
from typing import List

from experience import RolloutBuffer, ReplayBuffer


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


class CartpoleSoftActorCriticModel(nn.Module):
    def __init__(self):
        super(CartpoleSoftActorCriticModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(4, 128)
        self.pfc2 = nn.Linear(128, 2)
        self.q1fc = nn.Linear(128, 1)
        self.q2fc = nn.Linear(128, 1)

    def forward(self, x):
        core = nn.functional.relu(self.fc1(x))

        policy = self.pfc2(core)
        probs = torch.distributions.Categorical(logits=policy)
        action = probs.sample()

        q1 = self.q1fc(core)
        q2 = self.q2fc(core)

        return action, probs.log_prob(action), probs.entropy(), q1, q2

    def evaluate_value(self, x, actions):
        # flatten and do fully connected pass
        core = nn.functional.relu(self.fc1(x))

        policy = self.pfc2(core)
        probs = torch.distributions.Categorical(logits=policy)

        q1 = self.q1fc(core)
        q2 = self.q2fc(core)

        return probs.log_prob(actions).diag(), probs.entropy(), q1, q2


class MujoCoActorCriticModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size):
        super(MujoCoActorCriticModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(n_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.pfc2 = nn.Linear(hidden_size, n_outputs)
        self.vfc2 = nn.Linear(hidden_size, 1)
        self.logstd = nn.Parameter(torch.zeros(1, n_outputs))

    def forward(self, x):
        core = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x))))

        policy = self.pfc2(core)
        std = torch.exp(self.logstd)
        probs = torch.distributions.Normal(policy, std)
        action = probs.sample()
        value = self.vfc2(core)

        return action, probs.log_prob(action).sum(), probs.entropy(), value

    def evaluate_value(self, x, actions):
        # flatten and do fully connected pass
        core = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x))))

        policies = self.pfc2(core)
        std = torch.exp(self.logstd)
        
        log_probs = torch.zeros(len(policies))
        for i in range(len(policies)):
            probs = torch.distributions.Normal(policies[i], std)
            log_probs[i] = probs.log_prob(actions[i]).sum()

        value = self.vfc2(core)

        return log_probs, probs.entropy(), value


class MujoCoSoftActorCriticModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size):
        super(MujoCoSoftActorCriticModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(n_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.pfc2 = nn.Linear(hidden_size, n_outputs)
        self.q1fc2 = nn.Linear(hidden_size, 1)
        self.q2fc2 = nn.Linear(hidden_size, 1)
        self.logstd = nn.Parameter(torch.zeros(1, n_outputs))

    def forward(self, x):
        core = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x))))

        policy = self.pfc2(core)
        std = torch.exp(self.logstd)
        probs = torch.distributions.Normal(policy, std)
        action = probs.sample()
        q1 = self.q1fc2(core)
        q2 = self.q2fc2(core)

        return action, probs.log_prob(action).sum(), probs.entropy(), q1, q2

    def evaluate_value(self, x, actions):
        # flatten and do fully connected pass
        core = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x))))

        policies = self.pfc2(core)
        std = torch.exp(self.logstd)
        
        log_probs = torch.zeros(len(policies))
        for i in range(len(policies)):
            probs = torch.distributions.Normal(policies[i], std)
            log_probs[i] = probs.log_prob(actions[i]).sum()

        q1 = self.q1fc2(core)
        q2 = self.q2fc2(core)

        return log_probs, probs.entropy(), q1, q2


class MujoCoCriticModel(nn.Module):
    def __init__(self, n_inputs, hidden_size):
        super(MujoCoCriticModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(n_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.q1fc2 = nn.Linear(hidden_size, 1)

    def forward(self, *x):
        x = torch.cat(x, -1)
        core = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x))))

        q1 = self.q1fc2(core)
        
        return q1


class MujoCoActorModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size):
        super(MujoCoActorModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(n_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.pfc2 = nn.Linear(hidden_size, n_outputs)
        self.logstd = nn.Parameter(torch.zeros(1, n_outputs))

    def forward(self, x):
        core = nn.functional.relu(self.fc2(nn.functional.relu(self.fc1(x))))

        policy = self.pfc2(core)
        std = torch.exp(self.logstd)
        probs = torch.distributions.Normal(policy, std)
        u = probs.rsample()
        u_log_prob = probs.log_prob(u)
        a = torch.tanh(u)
        a_log_prob = u_log_prob - torch.log(1 - torch.square(a) + 1e-3)

        return a, a_log_prob.sum(-1, keepdim=True), probs.entropy()


class A2CAgent():
    def __init__(self, env: gym.Env, model: nn.Module, gamma=0.99, batch_size=128,
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
    def step(self, epsilon=0.0):
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
                 gamma=0.99, td_lambda=0.95, trajectory_length=2048, batch_size=64,
                 value_beta=0.5, entropy_beta=0.02, grad_limit=0.5, epsilon=0.2, writer=None):
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
    def step(self, epsilon=0.0):
        # run observation through network
        obs_tensor = torch.from_numpy(np.array([self.state])).float().to(self.device)
        action, log_prob, entropy, predicted_value = self.model(obs_tensor)

        # take action according to policy network
        action = action.reshape(self.env.action_space.shape)
        next_observation, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
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
                old_actions = torch.Tensor(old_actions).to(self.device).reshape(-1, self.env.action_space.shape[0])
                advantages = (advantages - advantages.mean() / (advantages.std() + 1e-8))
                advantages = torch.from_numpy(advantages).to(self.device)
                returns = torch.from_numpy(returns).to(self.device)
                old_log_probs = torch.from_numpy(old_log_probs).to(self.device)
                
                # make predicitions
                obs_tensor = torch.from_numpy(old_observations.reshape(-1, *self.env.observation_space.shape)).float().to(self.device)
                log_probs, entropy, predicted_values = self.model.evaluate_value(obs_tensor, old_actions)
                predicted_values = predicted_values.flatten()
                
                # compute loss

                # clip policy ratio loss
                ratio = torch.exp(log_probs.to(self.device) - old_log_probs)
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


class SACAgent():
    def __init__(self, env: gym.Env, actor: nn.Module,
                 critic1: nn.Module, critic2: nn.Module, critic1_target: nn.Module, critic2_target: nn.Module,
                 gamma=0.99, memory_size=int(1e6), batch_size=32, replay_start_size=1024,
                 value_beta=0.5, soft_update_rate=0.005, writer=None):
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for SACAgent Training")
        
        self.env = env
        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.critic1_target = critic1_target.to(self.device)
        self.critic2_target = critic2_target.to(self.device)
        
        self.soft_update(self.critic1, self.critic1_target, 1.0)
        self.soft_update(self.critic2, self.critic2_target, 1.0)

        self.gamma = gamma
        self.value_beta = value_beta
        self.soft_update_rate = soft_update_rate
        self.replay_start_size = replay_start_size

        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.value = 0.0
        self.final_values = collections.deque(maxlen=100)
        self.reset()
        self.final_values.clear()
        
        self.alpha = nn.Parameter(torch.tensor(0.2))
        self.memory = ReplayBuffer(self.memory_size)
    
    def reset(self):
        self.state, _ = self.env.reset()
        self.final_values.append(self.value)
        self.value = 0.0
    
    @torch.no_grad()
    def step(self, epsilon=0.0):
        # run observation through network
        obs_tensor = torch.from_numpy(np.array([self.state])).float().to(self.device)
        action, log_prob, entropy = self.actor(obs_tensor)

        # take action according to policy network
        action = action.reshape(self.env.action_space.shape)
        action = action.cpu().numpy()
        next_observation, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated

        # remember what happened for training
        self.value += reward
        exp = [self.state, action, reward, is_done, next_observation]
        self.memory.append(exp)

        self.state = next_observation
        self.last_episode_starts = is_done

        if is_done:
            self.reset()
    
    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def train(self, learning_rate, target_reward):
        best_mean_value = -10000000000
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        q1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=learning_rate)
        q2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=learning_rate)
        entropy_optimizer = torch.optim.Adam([self.alpha], lr=learning_rate)

        n = 1
        while True:
            self.step()
             
            # sync
            if len(self.memory) < self.replay_start_size:
                continue
            
            # train

            # prep data
            states, actions, rewards, dones, states_prime = self.memory.sample(self.batch_size)
            states = torch.from_numpy(states).float().to(self.device)
            states_prime = torch.from_numpy(states_prime).float().to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            done_mask = torch.BoolTensor(dones).int().to(self.device)
            
            # Q updates
            with torch.no_grad():
                next_actions, log_prob, entropy = self.actor(states_prime)
                log_prob = log_prob.to(self.device)
                q1 = self.critic1_target(states_prime, next_actions)
                q2 = self.critic2_target(states_prime, next_actions)
                next_q_vals = torch.min(q1, q2)
                v = (1 - done_mask) * (next_q_vals - self.alpha * log_prob.reshape(-1, 1)).squeeze(1)
                target_q_values = (rewards + self.gamma * v).float()
             
            q1 = self.critic1(states, actions).squeeze()
            q1_loss = nn.functional.mse_loss(q1, target_q_values)
            q1_optimizer.zero_grad()
            q1_loss.backward()
            q1_optimizer.step()

            q2 = self.critic2(states, actions).squeeze()
            q2_loss = nn.functional.mse_loss(q2, target_q_values)
            q2_optimizer.zero_grad()
            q2_loss.backward()
            q2_optimizer.step()

            # actor updates
            action, log_prob, entropy = self.actor(states)
            log_prob = log_prob.to(self.device)
            q1 = self.critic1(states, action)
            q2 = self.critic2(states, action)
            current_q_values = torch.min(q1, q2).squeeze(1)

            policy_loss = (self.alpha * log_prob - current_q_values).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            # update entropy
            entropy_optimizer.zero_grad()
            ent_coef_loss = (-self.alpha * (log_prob + self.target_entropy).detach()).mean()
            ent_coef_loss.backward()
            entropy_optimizer.step()
            
            # soft updates
            self.soft_update(self.critic1, self.critic1_target, self.soft_update_rate)
            self.soft_update(self.critic2, self.critic2_target, self.soft_update_rate)

            # logging
            mean_value = sum(self.final_values) / len(self.final_values) if len(self.final_values) != 0 else 0
            if mean_value != 0 and n % 1000 == 0:
                print(f"Step: {n} | Policy Loss: {policy_loss} | mean_value: {mean_value}")
                if self.writer is not None:
                    self.writer.add_scalar(f'{self.env.unwrapped.spec.id} SAC Mean Reward', mean_value, n)

                if mean_value >= best_mean_value:
                    best_mean_value = mean_value
                    torch.save(self.actor.state_dict(), f"./models/SAC_{self.env.unwrapped.spec.id}_best.pth")

                    if best_mean_value >= target_reward:
                        print("Solved!")
                        return
            n += 1


# simple unit test
if __name__ == "__main__":
    env = gym.make('Hopper-v4')

    state, _ = env.reset()
    model = MujoCoActorCriticModel(env.observation_space.shape[0], env.action_space.shape[0], 256)
    buffer = RolloutBuffer(4, env.observation_space, env.action_space, 1, 1.0, 1.0)

    observations = torch.zeros((2, *env.observation_space.shape))
    actions = torch.zeros((2, *env.action_space.shape))

    # act
    obs_tensor = torch.from_numpy(np.array([state])).float()
    action, log_prob, entropy, predicted_value = model(obs_tensor)
    actions[0] = action
    actionu = action.reshape(env.action_space.shape)
    next_observation, reward, terminated, truncated, _ = env.step(actionu.cpu().numpy())
    print(entropy)
    observations[0] = obs_tensor
    state = next_observation
    
    # act
    obs_tensor2 = torch.from_numpy(np.array([state])).float()
    action2, log_prob, entropy, predicted_value = model(obs_tensor2)
    print(entropy)
    observations[1] = obs_tensor2
    actions[1] = action2

    log_probs, entropy, predicted_values = model.evaluate_value(observations, actions)
    print(entropy)


