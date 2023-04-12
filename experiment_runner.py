import torch
from torch.utils.tensorboard import SummaryWriter
import gym
import dqn
import policygrad
import crossentropy
import actor_critic
import wrappers

import argparse
import time

if __name__ == "__main__":

    # load args
    parser = argparse.ArgumentParser(description="Train Or Run Deep Reinforcement Learning")
    parser.add_argument("env", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("agent", type=str)
    parser.add_argument("-r", "--target_reward", type=int, default=10)
    parser.add_argument("-l", "--load", type=str)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    args = parser.parse_args()
     
    # start tensorboard
    writer = SummaryWriter(f"logs/{args.env}-{args.model}-{args.agent}")

    # load env
    env = None
    mode = "human" if args.load is not None else None
    if args.env == "pong":
        env = gym.make("PongNoFrameskip-v4", render_mode=mode)
        env = wrappers.MaxAndSkipEnv(env)
        env = wrappers.FireResetEnv(env)
        env = wrappers.ProcessFrame84(env)
        env = wrappers.ImageToPyTorch(env)
        env = wrappers.ScaledFloatFrame(env)
    elif args.env == "space-invaders":
        pass
    elif args.env == "hopper":
        env = gym.make('Hopper-v4', render_mode=mode)
    elif args.env == "cartpole":
        env = gym.make('CartPole-v0', render_mode=mode)
    
    # load model
    model = None
    target = None
    if args.model == "cartpole-cross-entropy":
        model = crossentropy.CartpoleCrossEntropyModel()
    elif args.model == "cartpole-reinforce":
        model = policygrad.CartpoleREINFORCEModel()
    elif args.model == "cartpole-dqn":
        model = dqn.CartpoleDQNModel()
        target = dqn.CartpoleDQNModel()
    elif args.model == "cartpole-a2c":
        model = actor_critic.CartpoleActorCriticModel()
    elif args.model == "cartpole-sac":
        model = actor_critic.CartpoleSoftActorCriticModel()
    elif args.model == "image":
        model = dqn.ImageModel()
        target = dqn.ImageModel()
    elif args.model == "noisy-dueling-image":
        model = dqn.NoisyDuelingImageModel()
        target = dqn.NoisyDuelingImageModel()
    elif args.model == "image-a2c":
        model = actor_critic.ImageActorCriticModel()
    elif args.model == "mujoco-a2c":
        model = actor_critic.MujoCoActorCriticModel(env.observation_space.shape[0], env.action_space.shape[0], 256)
    elif args.model == "mujoco-sac":
        critic1 = actor_critic.MujoCoCriticModel(env.observation_space.shape[0] + env.action_space.shape[0], 256)
        critic2 = actor_critic.MujoCoCriticModel(env.observation_space.shape[0] + env.action_space.shape[0], 256)
        critic1_target = actor_critic.MujoCoCriticModel(env.observation_space.shape[0] + env.action_space.shape[0], 256)
        critic2_target = actor_critic.MujoCoCriticModel(env.observation_space.shape[0] + env.action_space.shape[0], 256)
        actor = actor_critic.MujoCoActorModel(env.observation_space.shape[0], env.action_space.shape[0], 256)

    # load algorithim
    agent = None
    if args.agent == "cross-entropy":
        agent = crossentropy.CrossEntropyAgent(env, model)
    elif args.agent == "reinforce":
        agent = policygrad.REINFORCEAgent(env, model, writer=writer)
    elif args.agent == "dqn":
        agent = dqn.DQNAgent(env, model, target, epsilon_duration=10**5, target_sync_time=1000, gamma=0.99, writer=writer)
    elif args.agent == "rainbow":
        agent = dqn.RainbowAgent(env, model, target, target_sync_time=1000, gamma=0.99, n_steps=3, writer=writer)
    elif args.agent == "a2c":
        agent = actor_critic.A2CAgent(env, model, gamma=0.99, writer=writer)
    elif args.agent == "ppo":
        agent = actor_critic.PPOAgent(env, model, writer=writer, batch_size=512)
    elif args.agent == "sac":
        agent = actor_critic.SACAgent(env, actor, critic1, critic2, critic1_target, critic2_target, writer=writer)

    # train or run
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))
        while True:
            value = agent.step(0.0)
            env.render()
            time.sleep(0.03)
    else:
        print(f"Training until we reach mean reward of {args.target_reward} with a learning rate of {args.learning_rate}")
        agent.train(args.learning_rate, args.target_reward)
