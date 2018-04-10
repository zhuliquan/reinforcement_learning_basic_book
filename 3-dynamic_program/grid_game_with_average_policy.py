#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/9
'''
这里面存在几个问题，按照书作者的意思是
作用的动作如果不可以移动的话，那么
用于迭代的 v(s') = v(s)
再用于计算 v(s) = Σπ*(r + γ*v(s'))
我感觉这个是不对
如果没有移动应该就没有 r 和 折扣这一说
但是这里但是姑且用这么书中的方式
之后的策略迭代和值迭代都是修改过的
'''
import pandas as pd
import numpy as np

class GridMDP():
    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.__action_dir = pd.Series(
            data = [np.array((-1, 0)),
                    np.array((1, 0)),
                    np.array((0, -1)),
                    np.array((0, 1))],
            index = self.action_space)
        self.terminal_states = [(0,0),(3,3)]
    def transform(self,state,action):
        dir = self.__action_dir[action]
        state_ = np.array(state) + dir
        if (state_ >= 0).all() and (state_ < 4).all():
            state_ = tuple(state_)
        else:
            state_ = state
        return state_

def average_policy(mdp, v_s,policy):
    state_space = mdp.state_space
    action_space = mdp.action_space
    reward = mdp.reward
    gamma = mdp.gamma
    while True:
        print(v_s)
        v_s_ = v_s.copy()
        for state in state_space:
            v_s_a = pd.Series()
            for action in action_space:
                state_ = mdp.transform(state,action)
                if state_ in mdp.terminal_states:
                    v_s_a[action] = 0
                elif state_ != state:
                    v_s_a[action] = v_s_[state_]
                else:
                    v_s_a[action] = v_s_[state]
            v_s[state] = sum([policy[action] * (reward + gamma * v_s_a[action])
                               for action in action_space])
        if (np.abs(v_s_ - v_s) < 1e-8).all():
            break

    return v_s

def main():
    state_space = [(i, j) for i in range(4) for j in range(4)]
    state_space.remove((0, 0))
    state_space.remove((3, 3))
    mdp = GridMDP(
        state_space=state_space,
        action_space=["n", "s", "w", "e"],
        reward=-1,
        gamma=1)
    v_s = pd.Series(np.zeros((len(state_space))),index=state_space)
    policy = pd.Series(data=0.25 * np.ones(shape=(4)), index=mdp.action_space)
    v_s = average_policy(mdp,v_s,policy)
    print("convergence valuation of __state is:")
    print(v_s)
if __name__ == '__main__':
    main()