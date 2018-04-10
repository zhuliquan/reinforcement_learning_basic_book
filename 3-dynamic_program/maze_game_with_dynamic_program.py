#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/9
import gym
import time
import pandas as pd
import numpy as np


def value_iterate(env):
    v_s = pd.Series(data=np.zeros(shape=len(env.states)), index=env.states)
    state_space = env.observation_space
    action_space = env.action_space
    policy = pd.Series(index=state_space)
    gamma = 0.8
    while True:
        print(v_s)
        v_s_ = v_s.copy()

        v_s_a = pd.Series()
        for action in action_space:
            state_,reward,is_done,_ = env.step(action)


        if (np.abs(v_s_ - v_s) < 1e-8).all():
            break
    return policy

def main():
    env = gym.make("MazeGame-v0")
    # policy = value_iterate(env)
    print(env.action_space)
    print(env.states)
    print("convergence policy is:")
    # print(policy)

if __name__ == '__main__':
    main()