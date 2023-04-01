import gym
import time
import torch
import torch.nn as nn
import numpy as np
import collections
import math


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)

        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)

        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)

        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias

        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * \
                self.epsilon_bias.data

        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight

        return nn.functional.linear(input, v, bias)


class CartpoleDQNModel(nn.Module):
    def __init__(self):
        super(CartpoleDQNModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # flatten and do fully connected pass
        out = nn.functional.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
               
        # define layers
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())
        
        self.conv_size = self._get_conv_out([1, 84, 84])
        self.fc1 = nn.Linear(self.conv_size, 512)
        self.fc2 = nn.Linear(512, 6)
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # flatten and do fully connected pass
        out = self.conv(x)
        out = out.view(x.size()[0], -1)
        out = nn.functional.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class NoisyDuelingImageModel(nn.Module):
    def __init__(self):
        super(NoisyDuelingImageModel, self).__init__()
               
        # define layers
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())
        
        self.conv_size = self._get_conv_out([1, 84, 84])

        self.fc_adv1 = NoisyLinear(self.conv_size, 256)
        self.fc_adv2 = NoisyLinear(256, 6)

        self.fc_val1 = NoisyLinear(self.conv_size, 256)
        self.fc_val2 = NoisyLinear(256, 1)
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # x = x.float() / 255
        out = self.conv(x)
        out = out.view(x.size()[0], -1)

        # flatten and do fully connected pass
        adv_out = self.fc_adv2(nn.functional.relu(self.fc_adv1(out)))
        val_out = self.fc_val2(nn.functional.relu(self.fc_val1(out)))
        return val_out + (adv_out - adv_out.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        return self.buffer.append(experience)

    def sample(self, sample_size):
        idxs = np.random.choice(len(self.buffer), sample_size, replace=False)

        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in idxs])

        return np.array(states), np.array(actions), np.array(rewards), np.array(dones, dtype=np.uint8), np.array(next_states)


class DQNAgent:
    def __init__(self, env: gym.Env, model: nn.Module, target_model: nn.Module, memory_size=100000, batch_size=32,
                 gamma=1.0, epsilon_start=1.0, epsilon_end=0.01, epsilon_duration=100000,
                 replay_start_size=10000, target_sync_time=1000, writer=None):
        
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        print(f"Using device: {self.device} for DQNAgent Training")
        
        # hyperparams
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_duration = epsilon_duration
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.target_sync_time = target_sync_time

        # models and state
        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.env = env
        self.memory = ReplayBuffer(memory_size)
        self.reset()
         
    def reset(self):
        self.state, _ = self.env.reset()
        self.value = 0.0
    
    def loss_from_memory(self):
        states, actions, rewards, dones, states_prime = self.memory.sample(self.batch_size)

        # GPU upload
        states = torch.from_numpy(states).to(self.device)
        states_prime = torch.from_numpy(states_prime).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)
        
        predicted_Q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # there is no target prediction in Q learning in the last step of an episode
        # without this training will never converge
        target_Q = self.target_model(states_prime).max(1)[0]
        target_Q[done_mask] = 0.0
        target_Q = target_Q.detach()

        expected_Q = (target_Q * self.gamma + rewards).float()

        return nn.MSELoss()(predicted_Q, expected_Q)

    def step(self, epsilon):
        episode_value = None

        # determine action via epsilon-greedy
        a = 0
        if np.random.random() < epsilon:
            a = self.env.action_space.sample()
        else:
            s = torch.from_numpy(np.array([self.state], copy=False)).to(self.device)
            q = self.model(s)
            _, a = torch.max(q, dim=1)
            a = int(a.item())
        
        # get s_prime, r from the env
        s_prime, r, terminated, truncated, _ = self.env.step(a)
        is_done = truncated or terminated
        self.value += r
        
        s_prime = s_prime

        # remember this step
        exp = [self.state, a, r, is_done, s_prime]
        self.memory.append(exp)
        
        # update state
        self.state = s_prime

        if is_done:
            episode_value = self.value
            self.reset()

        # if we finished an episode, let's report how well we did
        return episode_value
        
    def train(self, learning_rate, target_reward):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        count = 0
        ep_start_count = 0
        ep_start_time = time.time()
        final_rewards = []
        best_mean_reward = -10000000000
        while True:
            count += 1
            epsilon = max(self.epsilon_end, self.epsilon_start - (count / self.epsilon_duration))
            reward = self.step(epsilon)

            # if its the end of an episode, lets do some reporting and training
            if reward is not None:
                # stats storage
                final_rewards.append(reward)
                frame_rate = (count - ep_start_count) / (time.time() - ep_start_time)
                ep_start_count = 0
                ep_start_time = time.time()
                mean_reward = np.mean(final_rewards[-100:])
                
                # model saving
                if best_mean_reward < mean_reward:
                    best_mean_reward = mean_reward
                    torch.save(self.model.state_dict(), f"./models/DQN_{self.env.unwrapped.spec.id}_best.pth")
                    if best_mean_reward > target_reward:
                        print("solved!")
                        break
                
            # sync
            if len(self.memory) < self.replay_start_size:
                continue
            if count % self.target_sync_time == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"steps: {count} | episode: {len(final_rewards)} | average reward: {mean_reward} | epsilon: {epsilon} | framerate: {frame_rate}")

                # write mean reward to tensorboard
                if self.writer is not None:
                    self.writer.add_scalar('DQN Mean Reward', mean_reward, count)

            # training
            loss = self.loss_from_memory()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


# Has Double Q learning + n-step sampling (excludes categorical DQN which was in the paper)
# For true Rainbowness, you need to make your network have noisy layers & dueling (see NoisyDuelingImageModel for an example)
class RainbowAgent:
    def __init__(self, env: gym.Env, model: nn.Module, target_model: nn.Module, memory_size=100000, batch_size=32,
                 gamma=1.0, replay_start_size=10000, target_sync_time=1000, n_steps=3, writer=None):
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        print(f"Using device: {self.device} for RainbowAgent Training")
        
        # hyperparams
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size
        self.target_sync_time = target_sync_time
        self.n_steps = n_steps
        self.history_queue = collections.deque(maxlen=self.n_steps)

        # models and state
        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.env = env
        self.memory = ReplayBuffer(memory_size)
        self.reset()
    
    def reset(self):
        self.state, _ = self.env.reset()
        self.value = 0.0
    
    def loss_from_memory(self):
        states, actions, rewards, dones, states_prime = self.memory.sample(self.batch_size)

        # GPU upload
        states = torch.from_numpy(states).to(self.device)
        states_prime = torch.from_numpy(states_prime).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)
        
        predicted_Q = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            # double Q learning
            next_state_actions = self.model(states_prime).max(1)[1]
            next_state_actions = next_state_actions.unsqueeze(-1)

            # there is no target prediction in Q learning in the last step of an episode
            # without this training will never converge
            target_Q = self.target_model(states_prime).gather(1, next_state_actions).squeeze(-1)
            target_Q[done_mask] = 0.0
            target_Q = target_Q.detach()

            expected_Q = (target_Q * (self.gamma**self.n_steps) + rewards).float()

        return nn.MSELoss()(predicted_Q, expected_Q)
    
    @torch.no_grad()
    def step(self):
        episode_value = None

        # determine action
        s = torch.from_numpy(np.array([self.state], copy=False)).to(self.device)
        q = self.model(s)
        _, a = torch.max(q, dim=1)
        a = int(a.item())
        
        # get s_prime, r from the env
        s_prime, r, terminated, truncated, _ = self.env.step(a)
        is_done = truncated or terminated
        self.value += r
        
        s_prime = s_prime
        
        exp = [self.state, a, r, is_done, s_prime]
        self.history_queue.append(exp)
        
        # bellman unroll
        if len(self.history_queue) >= self.n_steps:
            # remember this unrolled nsteps
            rr = 0
            for h in self.history_queue:
                rr += h[2] * self.gamma
            self.memory.append([self.history_queue[0][0], self.history_queue[0][1], rr, is_done, s_prime])
        
        # update state
        self.state = s_prime

        if is_done:
            episode_value = self.value
            self.reset()
            self.history_queue.clear()

        # if we finished an episode, let's report how well we did
        return episode_value
    
    def train(self, learning_rate, target_reward):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        count = 0
        ep_start_count = 0
        ep_start_time = time.time()
        final_rewards = []
        best_mean_reward = -10000000000
        while True:
            count += 1
            reward = self.step()

            # if its the end of an episode, lets do some reporting and training
            if reward is not None:
                # stats storage
                final_rewards.append(reward)
                frame_rate = (count - ep_start_count) / (time.time() - ep_start_time)
                ep_start_count = 0
                ep_start_time = time.time()
                mean_reward = np.mean(final_rewards[-100:])
                
                # model saving
                if best_mean_reward < mean_reward:
                    best_mean_reward = mean_reward
                    torch.save(self.model.state_dict(), f"./models/Rainbow_{self.env.unwrapped.spec.id}_best.pth")
                    if best_mean_reward > target_reward:
                        print("solved!")
                        break
                
            # sync
            if len(self.memory) < self.replay_start_size:
                continue
            if count % self.target_sync_time == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                print(f"steps: {count} | episode: {len(final_rewards)} | average reward: {mean_reward} | framerate: {frame_rate}")
                
                # write mean reward to tensorboard
                if self.writer is not None:
                    self.writer.add_scalar('Rainbow Mean Reward', mean_reward, count)

            # training
            loss = self.loss_from_memory()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
