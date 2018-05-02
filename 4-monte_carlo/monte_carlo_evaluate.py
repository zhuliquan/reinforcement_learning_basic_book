#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/5/2
from collections import defaultdict
def mc(gamma,env,state_sample,reward_sample):
    V = defaultdict(float)
    N = defaultdict(int)
    states = env.observation_space
    num = len(state_sample)
    for i in range(num):
        G = 0.0
        episode_len = len(state_sample[i])
        # 从后往前尝试
        for episode in range(episode_len-1,-1,-1):
            G *= gamma
            G += reward_sample[i][episode]

        # 计算每一状态的值函数累加
        for episode in range(episode_len):
            s = state_sample[i][episode]
            V[s] += G
            N[s] += 1
            G    -= reward_sample[i][episode]
            G    /= gamma

    # 经验平均
    for s in states:
        if N[s] >= 0.000001:
            V[s] /= N[s]
    return V