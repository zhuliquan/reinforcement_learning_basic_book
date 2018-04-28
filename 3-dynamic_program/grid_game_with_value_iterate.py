#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2018/4/9
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
        self.terminal_space = [(0, 0), (3, 3)]
    def transform(self,state,action):
        dir = self.__action_dir[action]
        state_ = np.array(state) + dir
        if (state_ >= 0).all() and (state_ < 4).all():
            state_ = tuple(state_)
        else:
            state_ = state
        return state_

def value_iterate(mdp):
    state_space = mdp.state_space
    action_space = mdp.action_space
    gamma = mdp.gamma
    reward = mdp.reward
    v_s = pd.Series(data=np.zeros(shape=len(state_space)), index=state_space)
    policy = pd.Series(index=state_space)
    while True:
        print(v_s)
        v_s_ = v_s.copy()
        for state in state_space:
            v_s_a = pd.Series()
            for action in action_space:
                state_ = mdp.transform(state,action)
                if state_ in mdp.terminal_space:
                    v_s_a[action] = reward
                else:
                    v_s_a[action] = reward + gamma * v_s_[state_]

            v_s[state] = v_s_a.max()
            policy[state] = np.random.choice(v_s_a[v_s_a == v_s[state]].index)

        if (np.abs(v_s_ - v_s) < 1e-8).all():
            break
    return policy

def main():
    state_space = [(i, j) for i in range(4) for j in range(4)]
    state_space.remove((0,0))
    state_space.remove((3,3))
    mdp = GridMDP(
        state_space  = state_space,
        action_space = ["n","s","w","e"],
        reward  = -1,
        gamma    = 0.9)
    policy = value_iterate(mdp)
    print("convergence policy is:")
    print(policy)

if __name__ == '__main__':
    main()