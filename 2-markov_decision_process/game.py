#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/7
import gym
def main():
    env = gym.make("MazeGame-v0")
    env.reset()
    env.render()
if __name__ == '__main__':
    main()