import numpy as np
import gym
from discrete_agents import DeepQLearning
from discrete_agents import PolicyGradient
from discrete_agents import ActorCritic
from continuous_agents import TD3

if __name__ == '__main__':
    # env_ = gym.make('CartPole-v1')
    # agent = ActorCritic(env_, [16, 16, 32, 32], [16, 16, 32, 32])
    # agent.fit()
    env_ = gym.make('Pendulum-v1')
    agent = TD3(env_, [16, 16, 32, 32], [16, 16, 32, 32])
    agent.fit()
