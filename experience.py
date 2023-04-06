import numpy as np
import gym
import torch
import collections
from typing import Optional
import wrappers

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

    def trajectory_sample(self, sample_size):
        idxs = list(range(len(self) - sample_size, len(self)))

        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in idxs])

        return np.array(states), np.array(actions), np.array(rewards), np.array(dones, dtype=np.uint8), np.array(next_states)


# heavily inspiried by the actual stable baselines implementation
class RolloutBuffer:
    def __init__(self, capacity: int, observation_space: gym.spaces.Space, action_space: gym.spaces.Space,
                 num_envs: int = 1, gamma: float = 0.99, td_lambda: float = 1.0):
        self.capacity = capacity
        self.obs_shape = observation_space.shape
        self.action_dim = int(np.prod(action_space.shape))
        self.n_envs = num_envs
        self.gamma = gamma
        self.td_lambda = td_lambda
        self.clear()

    def clear(self):
        self.observations = np.zeros((self.capacity, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.capacity, self.n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    def append(self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray, episode_start: np.ndarray, value: torch.Tensor, log_prob: torch.Tensor):
        assert not self.full
        self.observations[self.pos] = np.array(observation).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1

        if self.pos == self.capacity:
            self.full = True
    
    # Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: np.ndarray):
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0
        for step in reversed(range(self.capacity)):
            if step == self.capacity - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step].item() + self.gamma * next_values * next_non_terminal - self.values[step].item()
            last_gae_lam = delta + self.gamma * self.td_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values

    def _get_samples(self, batch_indices: np.ndarray):
        return self.observations[batch_indices], self.actions[batch_indices], self.values[batch_indices].flatten(),\
            self.log_probs[batch_indices].flatten(), self.advantages[batch_indices].flatten(), self.returns[batch_indices].flatten()

    def get(self, batch_size: Optional[int] = None):
        if not self.full:
            print("ERROR: RolloutBuffer is not full but you called get() on it!")
        
        # indices = np.random.permutation(self.capacity * self.n_envs)
        indices = list(range(self.capacity))

        # return full rollout if None
        if batch_size is None:
            batch_size = self.capacity

        start_idx = 0
        while start_idx < self.capacity * self.n_envs:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size


# simple unit test
if __name__ == "__main__":
    env = gym.make('PongNoFrameskip-v0')
    env = wrappers.MaxAndSkipEnv(env)
    env = wrappers.FireResetEnv(env)
    env = wrappers.ProcessFrame84(env)
    env = wrappers.ImageToPyTorch(env)
    env = wrappers.ScaledFloatFrame(env)

    buffer = RolloutBuffer(4, env.observation_space, env.action_space, 1, 1.0, 1.0)
    state, _ = env.reset()
    buffer.append(state, torch.Tensor([1]), 1.0, np.ones((1), dtype=bool), torch.Tensor([2.0]), torch.Tensor([0.9]))
    buffer.append(state, torch.Tensor([1]), 1.0, np.zeros((1), dtype=bool), torch.Tensor([4.0]), torch.Tensor([0.9]))
    buffer.append(state, torch.Tensor([1]), 1.0, np.zeros((1), dtype=bool), torch.Tensor([16.0]), torch.Tensor([0.9]))
    buffer.append(state, torch.Tensor([1]), 1.0, np.zeros((1), dtype=bool), torch.Tensor([7.0]), torch.Tensor([0.9]))
    buffer.compute_returns_and_advantage(torch.Tensor([3.0]), True)

    print(state)
    for old_observations, old_actions, old_values, old_log_probs, advantages, returns in buffer.get():
        print(old_observations)




