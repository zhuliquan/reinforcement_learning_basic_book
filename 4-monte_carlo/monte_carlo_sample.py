#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/30
import numpy as np
def gen_andom(env,num):
    state_sample = []
    action_sample = []
    reward_sample = []
    # 模拟num次的采样
    for i in range(num):
        s_tmp = []
        a_tmp = []
        r_tmp = []
        s = env.reset()
        is_done = False
        # 每次采样的过程
        while not is_done:
            a = np.random.choice(env.action_space)
            s_,r,is_done,_ = env.transform(s,a)
            s_tmp.append(s)
            a_tmp.append(a)
            r_tmp.append(r)
            s = s_
        state_sample.append(s_tmp)
        action_sample.append(a_tmp)
        reward_sample.append(r_tmp)
    return state_sample,action_sample,reward_sample




