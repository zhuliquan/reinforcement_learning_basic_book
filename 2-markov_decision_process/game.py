#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/7
import gym
import numpy as np
def main():
    env = gym.make("MazeGame-v0")
    s  = env.reset()
    a_s = env.action_space
    for i in range(100):
        env.render()
        a = np.random.choice(a_s)
        print(a)
        s,r,t,_ = env.step(a)

if __name__ == '__main__':
    main()