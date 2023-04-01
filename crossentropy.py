import gym
import torch
import torch.nn as nn
import numpy as np


class CartpoleCrossEntropyModel(nn.Module):
    def __init__(self):
        super(CartpoleCrossEntropyModel, self).__init__()
        
        # define layers
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        # flatten and do fully connected pass
        out = nn.functional.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class CrossEntropyAgent():
    def __init__(self, env: gym.Env, model: nn.Module):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for CrossEntropyAgent Training")

        self.model = model.to(self.device)
        self.env = env
    
    def iterate_batch(self, batch_size):
        batch = []
        value = 0
        episode_steps = []
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
            value += reward
            episode_steps.append((observation, action))

            if is_done:
                episode = (value, episode_steps)
                batch.append(episode)
                value = 0
                episode_steps = []
                next_observation, _ = self.env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []

            observation = next_observation

    def filter_batch(self, batch, percentile):
        # compute value stastics and cutoff
        values = list(map(lambda b: b[0], batch))

        value_threshold = np.percentile(values, percentile)
        mean_value = float(np.mean(values))

        # filter the batch
        train_observations = []
        train_actions = []
        for value, step in batch:
            if value >= value_threshold:
                train_observations.extend(map(lambda s: s[0], step))
                train_actions.extend(map(lambda s: s[1], step))

        return torch.Tensor(train_observations), torch.Tensor(train_actions), value_threshold, mean_value

    def train(self, learning_rate, desired_mean, batch_size=16, percentile=70):
        best_mean_value = -10000000000
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for n, batch in enumerate(self.iterate_batch(batch_size)):
            obvs, actions, value_threshold, mean_value = self.filter_batch(batch, percentile)
            # move to GPU
            obvs = obvs.to(self.device)
            actions = actions.long().to(self.device)

            action_scores = self.model(obvs)
            loss = criterion(action_scores, actions)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Batch: {n} | Loss: {loss} | mean_value: {mean_value}")
            if mean_value >= best_mean_value:
                best_mean_value = mean_value
                torch.save(self.model.state_dict(), f"./models/CrossEntropy_{self.env.unwrapped.spec.id}_best.pth")

            if mean_value >= desired_mean:
                print("Finished Training")
                return


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    model = CartpoleCrossEntropyModel()
    agent = CrossEntropyAgent(env, model)
    agent.train(0.01, 200, batch_size=32)
