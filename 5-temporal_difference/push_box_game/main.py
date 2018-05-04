#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2017/10/4
import gym
from agent import QLearningAgent
if __name__ == "__main__":
    env = gym.make("PushBoxGame-v0")
    RL = QLearningAgent(actions=env.action_space)
    RL.train(env)