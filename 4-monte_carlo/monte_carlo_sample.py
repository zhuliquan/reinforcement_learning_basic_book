#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/30
import numpy as np
def gen_andom(env,num):
    state_sample = []
    action_sample = []
    reward_sample = []
    for i in range(num):
        s_tmp = []
        a_tmp = []
        r_tmp = []
        s = env.reset()
        is_done = False
        while not is_done:
            a = np.random.choice(env.action_space)
            s_,r,t,_ = env.step(a)
            s_tmp.append(s)
            a_tmp.append(a)


