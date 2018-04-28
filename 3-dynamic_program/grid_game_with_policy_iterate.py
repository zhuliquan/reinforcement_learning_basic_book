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
        self.terminal_space = [(0,0),(3,3)]
    def transform(self,state,action):
        dir = self.__action_dir[action]
        state_ = np.array(state) + dir
        if (state_ >= 0).all() and (state_ < 4).all():
            state_ = tuple(state_)
        else:
            state_ = state
        return state_

def policy_evaluate(v_s,policy,mdp):
    state_space = mdp.state_space
    gamma   = mdp.gamma
    reward = mdp.reward
    while True:
        v_s_ = v_s.copy()
        for state in state_space:
            action = policy[state]
            state_ = mdp.transform(state,action)
            if state_ in mdp.terminal_space: # 发生转移
                v_s[state] = reward + 0.0
            elif state_ != state:       # 终点位置
                v_s[state] = reward + gamma * v_s_[state_]
            else:                       # 没有发生转移
                v_s[state] = reward + gamma * v_s_[state_]

        if (np.abs(v_s_ - v_s) < 1e-8).all():
            break
    return v_s

def policy_improve(v_s,mdp):
    state_space = mdp.state_space
    action_space = mdp.action_space
    gamma   = mdp.gamma
    reward = mdp.reward
    policy_ = pd.Series(index=state_space)
    for state in state_space:
        v_s_a = pd.Series()
        for action in action_space:
            state_ = mdp.transform(state,action)
            if state_ in mdp.terminal_space:
                v_s_a[action] = reward
            else:
                v_s_a[action] = reward + gamma *  v_s[state_]

        # 随机选取最大的值
        m = v_s_a.max()
        policy_[state] = np.random.choice(v_s_a[v_s_a == m].index)
    return policy_

def policy_iterate(mdp):
    v_s = pd.Series(data=np.zeros(shape=len(mdp.state_space)), index=mdp.state_space)
    policy = pd.Series(data = np.random.choice(mdp.action_space,size=(len(mdp.state_space))),index = mdp.state_space)
    while True:
        print(policy)
        v_s = policy_evaluate(v_s, policy, mdp)
        policy_ = policy_improve(v_s,mdp)
        if (policy_ == policy).all():
            break
        else:
            policy = policy_
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
    policy = policy_iterate(mdp)
    print("convergence policy is:")
    print(policy)

if __name__ == '__main__':
    main()