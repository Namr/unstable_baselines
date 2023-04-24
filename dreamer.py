import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from experience import ReplayBuffer
from attrdict import AttrDict


class DynamicInfos:
    def __init__(self, device):
        self.data = {}

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def get_stacked(self, time_axis=1):
        stacked_data = AttrDict(
            {
                key: torch.stack(self.data[key], dim=time_axis).cuda()
                for key in self.data
            }
        )
        self.clear()
        return stacked_data

    def clear(self):
        self.data = {}


def horizontal_forward(network, x, y=None, input_shape=(-1,), output_shape=(-1,)):
    batch_with_horizon_shape = x.shape[: -len(input_shape)]
    if not batch_with_horizon_shape:
        batch_with_horizon_shape = (1,)
    if y is not None:
        x = torch.cat((x, y), -1)
        input_shape = (x.shape[-1],)  #
    x = x.reshape(-1, *input_shape)
    x = network(x)

    x = x.reshape(*batch_with_horizon_shape, *output_shape)
    return x


def build_network(input_size, hidden_size, num_layers, activation, output_size):
    assert num_layers >= 2, "num_layers must be at least 2"
    activation = getattr(nn, activation)()
    layers = []
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(activation)

    for i in range(num_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(activation)

    layers.append(nn.Linear(hidden_size, output_size))

    network = nn.Sequential(*layers)
    network.apply(initialize_weights)
    return network


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def create_normal_dist(
    x,
    std=None,
    mean_scale=1,
    init_std=0,
    min_std=0.1,
    activation=None,
    event_shape=None,
):
    if std == None:
        mean, std = torch.chunk(x, 2, -1)
        mean = mean / mean_scale
        if activation:
            mean = activation(mean)
        mean = mean_scale * mean
        std = F.softplus(std + init_std) + min_std
    else:
        mean = x
    dist = torch.distributions.Normal(mean, std)
    if event_shape:
        dist = torch.distributions.Independent(dist, event_shape)
    return dist


def compute_lambda_values(rewards, values, continues, horizon_length, device, lambda_):
    """
    rewards : (batch_size, time_step, hidden_size)
    values : (batch_size, time_step, hidden_size)
    continue flag will be added
    """
    rewards = rewards[:, :-1]
    continues = continues[:, :-1]
    next_values = values[:, 1:]
    last = next_values[:, -1]
    inputs = rewards + continues * next_values * (1 - lambda_)

    outputs = []
    # single step
    for index in reversed(range(horizon_length - 1)):
        last = inputs[:, index] + continues[:, index] * lambda_ * last
        outputs.append(last)
    returns = torch.stack(list(reversed(outputs)), dim=1).to(device)
    return returns


class Encoder(nn.Module):
    def __init__(self, observation_shape, depth, kernel_size, stride):
        super().__init__()

        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Conv2d(
                self.observation_shape[0],
                depth,
                kernel_size,
                stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                depth * 1,
                depth * 2,
                kernel_size,
                stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                depth * 2,
                depth * 4,
                kernel_size,
                stride,
            ),
            nn.ReLU(),
            nn.Conv2d(
                depth * 4,
                depth * 8,
                kernel_size,
                stride,
            ),
            nn.ReLU(),
        )
        self.network.apply(initialize_weights)

    def forward(self, x):
        return horizontal_forward(self.network, x, input_shape=self.observation_shape)


class Decoder(nn.Module):
    def __init__(self, observation_shape, stochastic_size, deterministic_size, depth, kernel_size, stride):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.observation_shape = observation_shape

        self.network = nn.Sequential(
            nn.Linear(
                self.deterministic_size + self.stochastic_size, depth * 32
            ),
            nn.Unflatten(1, (depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(
                depth * 32,
                depth * 4,
                kernel_size,
                stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                depth * 4,
                depth * 2,
                kernel_size,
                stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                depth * 2,
                depth * 1,
                kernel_size + 1,
                stride,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                depth * 1,
                self.observation_shape[0],
                kernel_size + 1,
                stride,
            ),
        )
        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=self.observation_shape
        )
        dist = create_normal_dist(x, std=1, event_shape=len(self.observation_shape))
        return dist
   

class RSSM(nn.Module):
    def __init__(self, action_size, state_size, stochastic_size, deterministic_size, hidden_size, num_layers):
        super().__init__()

        self.recurrent_model = RecurrentModel(action_size, stochastic_size, deterministic_size, hidden_size)
        self.transition_model = TransitionModel(stochastic_size, deterministic_size, hidden_size, num_layers)
        self.representation_model = RepresentationModel(state_size, stochastic_size, deterministic_size, hidden_size, num_layers)

    def recurrent_model_input_init(self, batch_size):
        return self.transition_model.input_init(
            batch_size
        ), self.recurrent_model.input_init(batch_size)


class RecurrentModel(nn.Module):
    def __init__(self, action_size, stochastic_size, deterministic_size, hidden_size):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.activation = "ELU"

        self.linear = nn.Linear(
            self.stochastic_size + action_size, hidden_size
        )
        self.recurrent = nn.GRUCell(hidden_size, self.deterministic_size)

    def forward(self, embedded_state, action, deterministic):
        x = torch.cat((embedded_state, action), 1)
        x = self.activation(self.linear(x))
        x = self.recurrent(x, deterministic)
        return x

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.deterministic_size).cuda()


class TransitionModel(nn.Module):
    def __init__(self, stochastic_size, deterministic_size, hidden_size, num_layers):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.network = build_network(
            self.deterministic_size,
            hidden_size,
            num_layers,
            "ELU",
            stochastic_size * 2,
        )

    def forward(self, x):
        x = self.network(x)
        prior_dist = create_normal_dist(x, min_std=0.1)
        prior = prior_dist.rsample()
        return prior_dist, prior

    def input_init(self, batch_size):
        return torch.zeros(batch_size, self.stochastic_size).cuda()


class RepresentationModel(nn.Module):
    def __init__(self, state_size, stochastic_size, deterministic_size, hidden_size, num_layers):
        super().__init__()
        self.embedded_state_size = state_size
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.network = build_network(
            self.embedded_state_size + self.deterministic_size,
            hidden_size,
            num_layers,
            "ELU",
            self.stochastic_size * 2,
        )

    def forward(self, embedded_observation, deterministic):
        x = self.network(torch.cat((embedded_observation, deterministic), 1))
        posterior_dist = create_normal_dist(x, min_std=0.1)
        posterior = posterior_dist.rsample()
        return posterior_dist, posterior


class RewardModel(nn.Module):
    def __init__(self, stochastic_size, deterministic_size, hidden_size, num_layers):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size

        self.network = build_network(
            self.stochastic_size + self.deterministic_size,
            hidden_size,
            num_layers,
            "ELU",
            1,
        )

    def forward(self, posterior, deterministic):
        x = horizontal_forward(
            self.network, posterior, deterministic, output_shape=(1,)
        )
        dist = create_normal_dist(x, std=1, event_shape=1)
        return dist


class DreamerAgent():
    def __init__(self, env: gym.Env,
                 actor: nn.Module, critic: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 rssm: nn.Module, reward_model: nn.Module,
                 gamma=0.99, memory_size=int(1e6), batch_size=32, replay_start_size=1024, writer=None):
        
        self.writer = writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} for DreamerAgent Training")

        self.env = env
        self.action_size = env.action_space.shape[0]
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.rssm = rssm.to(self.device)
        self.reward_predictor = reward_model.to(self.device)

        self.gamma = gamma
        self.replay_start_size = replay_start_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = ReplayBuffer(self.memory_size)
        self.num_total_episode = 0

        self.dynamic_learning_infos = DynamicInfos(self.device)
        self.behavior_learning_infos = DynamicInfos(self.device)
    
    # TODO: this should be remade into the step format that other agents use
    @torch.no_grad()
    def collect_exp(self, eps_to_collect, epsilon=0.0, training=True):
        for epi in range(eps_to_collect):
            posterior, deterministic = self.rssm.recurrent_model_input_init(1)
            action = torch.zeros(1, self.action_size).to(self.device)

            observation, _ = self.env.reset()
            embedded_observation = self.encoder(
                torch.from_numpy(observation).float().to(self.device)
            )

            score = 0
            score_lst = np.array([])
            done = False

            while not done:
                deterministic = self.rssm.recurrent_model(
                    posterior, action, deterministic
                )
                embedded_observation = embedded_observation.reshape(1, -1)
                _, posterior = self.rssm.representation_model(
                    embedded_observation, deterministic
                )
                action = self.actor(posterior, deterministic).detach()

                if self.discrete_action_bool:
                    buffer_action = action.cpu().numpy()
                    env_action = buffer_action.argmax()

                else:
                    buffer_action = action.cpu().numpy()[0]
                    env_action = buffer_action

                next_observation, reward, done, info = self.env.step(env_action)
                exp = [observation, buffer_action, reward, done, next_observation]
                self.memory.append(exp)

                score += reward
                embedded_observation = self.encoder(torch.from_numpy(next_observation).float().to(self.device))
                observation = next_observation

                if done and training:
                    self.num_total_episode += 1
                    self.writer.add_scalar("dreamer training reward", score, self.num_total_episode)
                    break
                elif done and not training:
                    score_lst = np.append(score_lst, score)
                    break

            if not training:
                avg_score = score_lst.mean()
                print("average reward: ", avg_score)
                self.writer.add_scalar("dreamer test reward", avg_score, self.num_total_episode)
                return avg_score

    def train(self, learning_rate, target_reward):
        # optimizer
        model_params = (list(self.encoder.parameters()) + list(self.decoder.parameters()) +
                        list(self.rssm.parameters()) + list(self.reward_predictor.parameters()))

        self.model_optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
    
        # seed memory buffer
        self.collect_exp(20)

        while True:
            for interval in range(100):
                # collect sample and put on GPU
                states, actions, rewards, dones, states_prime = self.memory.sample(self.batch_size)
                states = torch.from_numpy(states).float().to(self.device)
                states_prime = torch.from_numpy(states_prime).float().to(self.device)
                actions = torch.from_numpy(actions).to(self.device)
                rewards = torch.from_numpy(rewards).to(self.device)
                dones = torch.BoolTensor(dones).int().to(self.device)
                
                # train model and actor/critic
                posteriors, deterministics = self.dynamic_learning(states, actions, rewards, dones, states_prime)
                self.behavior_learning(posteriors, deterministics)

            # collect more exp and evaluate progress
            self.collect_exp(1)
            self.collect_exp(3, training=False)
    
    def dynamic_learning(self, states, actions, rewards, dones, states_prime):
        prior, deterministic = self.rssm.recurrent_model_input_init(len(actions))
        embedded_observation = self.encoder(states)

        for t in range(1, self.config.batch_length):
            deterministic = self.rssm.recurrent_model(
                prior, actions[:, t - 1], deterministic
            )
            prior_dist, prior = self.rssm.transition_model(deterministic)
            posterior_dist, posterior = self.rssm.representation_model(
                embedded_observation[:, t], deterministic
            )

            self.dynamic_learning_infos.append(
                priors=prior,
                prior_dist_means=prior_dist.mean,
                prior_dist_stds=prior_dist.scale,
                posteriors=posterior,
                posterior_dist_means=posterior_dist.mean,
                posterior_dist_stds=posterior_dist.scale,
                deterministics=deterministic,
            )

            prior = posterior

        infos = self.dynamic_learning_infos.get_stacked()
        self._model_update(states, actions, rewards, dones, states_prime, infos)
        return infos.posteriors.detach(), infos.deterministics.detach()

    def _model_update(self, states, actions, rewards, dones, states_prime, posterior_info):
        # compute KL divergence on model
        reconstructed_observation_dist = self.decoder(posterior_info.posteriors, posterior_info.deterministics)
        reconstruction_observation_loss = reconstructed_observation_dist.log_prob(states[:, 1:])
        reward_dist = self.reward_predictor(posterior_info.posteriors, posterior_info.deterministics)
        reward_loss = reward_dist.log_prob(rewards[:, 1:])

        prior_dist = create_normal_dist(posterior_info.prior_dist_means, posterior_info.prior_dist_stds, event_shape=1)
        posterior_dist = create_normal_dist(posterior_info.posterior_dist_means, posterior_info.posterior_dist_stds, event_shape=1)

        kl_divergence_loss = torch.mean(torch.distributions.kl.kl_divergence(posterior_dist, prior_dist))
        kl_divergence_loss = torch.max(torch.tensor(self.config.free_nats).to(self.device), kl_divergence_loss)

        model_loss = (self.config.kl_divergence_scale * kl_divergence_loss - reconstruction_observation_loss.mean() - reward_loss.mean())
        
        # optimize
        self.model_optimizer.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model_params,
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.model_optimizer.step()

    def behavior_learning(self, states, deterministics):
        state = states.reshape(-1, self.config.stochastic_size)
        deterministic = deterministics.reshape(-1, self.config.deterministic_size)

        # continue_predictor reinit
        for t in range(15):
            action = self.actor(state, deterministic)
            deterministic = self.rssm.recurrent_model(state, action, deterministic)
            _, state = self.rssm.transition_model(deterministic)
            self.behavior_learning_infos.append(
                priors=state, deterministics=deterministic
            )

        self._agent_update(self.behavior_learning_infos.get_stacked())

    def _agent_update(self, behavior_learning_infos):
        predicted_rewards = self.reward_predictor(behavior_learning_infos.priors, behavior_learning_infos.deterministics).mean
        values = self.critic(behavior_learning_infos.priors, behavior_learning_infos.deterministics).mean

        continues = self.config.discount * torch.ones_like(values)

        lambda_values = compute_lambda_values(
            predicted_rewards,
            values,
            continues,
            self.config.horizon_length,
            self.device,
            self.config.lambda_,
        )

        actor_loss = -torch.mean(lambda_values)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.actor_optimizer.step()

        value_dist = self.critic(
            behavior_learning_infos.priors.detach()[:, :-1],
            behavior_learning_infos.deterministics.detach()[:, :-1],
        )
        value_loss = -torch.mean(value_dist.log_prob(lambda_values.detach()))

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.clip_grad,
            norm_type=self.config.grad_norm_type,
        )
        self.critic_optimizer.step()

