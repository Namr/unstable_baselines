import gym
import time
import torch
import torch.nn as nn
import numpy as np
import collections
import math


class CartpoleREINFORCEModel(nn.Module):
    def __init__(self):
        super(CartpoleREINFORCEModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # flatten and do fully connected pass
        out = nn.functional.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class REINFORCEAgent():
    def __init__(self, env: gym.Env, model: nn.Module, gamma=1.0, batch_size=32, writer=None):
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for REINFORCEAgent Training")

        self.env = env
        self.model = model.to(self.device)
        self.gamma = gamma
        self.batch_size = batch_size

    def iterate_batch(self):
        batch_states = []
        batch_actions = []
        batch_values = []
        rewards = []
        final_values = []
        episodes = 0

        observation, _ = self.env.reset()
        sm = nn.Softmax(dim=1)
        
        while True:
            # run observation through network
            obs_tensor = torch.from_numpy(np.array([observation])).to(self.device)
            policy = sm(self.model(obs_tensor).to('cpu'))
            policy = policy.data.numpy()[0]

            # choose action according to the policy
            action = np.random.choice(len(policy), p=policy)
            next_observation, reward, terminated, truncated, _ = self.env.step(action)
            is_done = terminated or truncated

            # store what just happened
            batch_states.append(observation)
            batch_actions.append(action)
            rewards.append(reward)

            if is_done:
                # compute values from rewards in an episode
                values = []
                value = 0
                for r in reversed(rewards):
                    value *= self.gamma
                    value += r
                    values.append(value)
                batch_values.extend(reversed(values))
                final_values.append(value)

                # reset env
                rewards = []
                next_observation, _ = self.env.reset()
                episodes += 1

            if episodes == self.batch_size:
                yield batch_states, batch_actions, batch_values, sum(final_values) / len(final_values)
                episodes = 0
                batch_states = []
                batch_actions = []
                batch_values = []
                rewards = []
                final_values = []

            observation = next_observation

    def train(self, learning_rate, target_reward):
        best_mean_value = -10000000000
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        n = 1
        for states, actions, values, mean_value in self.iterate_batch():
            states = torch.Tensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            values = torch.Tensor(values).to(self.device)

            action_scores = self.model(states)
            action_probabilities = nn.functional.log_softmax(action_scores, dim=1)
            scaled_action_probs = values * action_probabilities[range(len(states)), actions]
            loss = -scaled_action_probs.mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # logging
            print(f"Batch: {n} | Loss: {loss} | mean_value: {mean_value}")
            if self.writer is not None:
                self.writer.add_scalar('REINFORCE Mean Reward', mean_value, n * self.batch_size)

            if mean_value >= best_mean_value:
                best_mean_value = mean_value
                torch.save(self.model.state_dict(), f"./models/REINFORCE_{self.env.unwrapped.spec.id}_best.pth")

                if best_mean_value >= target_reward:
                    print("Solved!")
                    return
            n += 1




