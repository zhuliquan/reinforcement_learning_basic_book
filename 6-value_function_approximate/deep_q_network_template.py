#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author : zlq16
# date   : 2017/10/31
import tensorflow as tf
import cv2
import random
import numpy as np
from collections import deque

#Game的定义类，此处Game是什么不重要，只要提供执行Action的方法，获取当前游戏区域像素的方法即可
class Game(object):
    def __init__(self):  #Game初始化
    # action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
    # ai控制棒子左右移动；返回游戏界面像素数和对应的奖励。(像素->奖励->强化棒子往奖励高的方向移动)
        pass
    def step(self, action):
        pass
# learning_rate
GAMMA = 0.99
# 跟新梯度
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# 测试观测次数
EXPLORE = 500000
OBSERVE = 500
# 记忆经验大小
REPLAY_MEMORY = 500000
# 每次训练取出的记录数
BATCH = 100
# 输出层神经元数。代表3种操作-MOVE_STAY:[1, 0, 0]  MOVE_LEFT:[0, 1, 0]  MOVE_RIGHT:[0, 0, 1]
output = 3
MOVE_STAY =[1, 0, 0]
MOVE_LEFT =[0, 1, 0]
MOVE_RIGHT=[0, 0, 1]
input_image = tf.placeholder("float", [None, 80, 100, 4])  # 游戏像素
action = tf.placeholder("float", [None, output])           # 操作

#定义CNN-卷积神经网络
def convolutional_neural_network(input_image):
    weights = {'w_conv1':tf.Variable(tf.zeros([8, 8, 4, 32])),
               'w_conv2':tf.Variable(tf.zeros([4, 4, 32, 64])),
               'w_conv3':tf.Variable(tf.zeros([3, 3, 64, 64])),
               'w_fc4':tf.Variable(tf.zeros([3456, 784])),
               'w_out':tf.Variable(tf.zeros([784, output]))}

    biases = {'b_conv1':tf.Variable(tf.zeros([32])),
              'b_conv2':tf.Variable(tf.zeros([64])),
              'b_conv3':tf.Variable(tf.zeros([64])),
              'b_fc4':tf.Variable(tf.zeros([784])),
              'b_out':tf.Variable(tf.zeros([output]))}

    conv1 = tf.nn.relu(tf.nn.conv2d(input_image, weights['w_conv1'], strides = [1, 4, 4, 1], padding = "VALID") + biases['b_conv1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights['w_conv2'], strides = [1, 2, 2, 1], padding = "VALID") + biases['b_conv2'])
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, weights['w_conv3'], strides = [1, 1, 1, 1], padding = "VALID") + biases['b_conv3'])
    conv3_flat = tf.reshape(conv3, [-1, 3456])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, weights['w_fc4']) + biases['b_fc4'])

    output_layer = tf.matmul(fc4, weights['w_out']) + biases['b_out']
    return output_layer

#训练神经网络
def train_neural_network(input_image):
    argmax = tf.placeholder("float", [None, output])
    gt = tf.placeholder("float", [None])

    #损失函数
    predict_action = convolutional_neural_network(input_image)
    action = tf.reduce_sum(tf.mul(predict_action, argmax), reduction_indices = 1) #max(Q(S,:))
    cost = tf.reduce_mean(tf.square(action - gt))
    optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)

    #游戏开始
    game = Game()
    D = deque()
    _, image = game.step(MOVE_STAY)
    image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    input_image_data = np.stack((image, image, image, image), axis = 2)

    with tf.Session() as sess:
        #初始化神经网络各种参数
        sess.run(tf.initialize_all_variables())
        #保存神经网络参数的模块
        saver = tf.train.Saver()

        #总的运行次数
        n = 0
        epsilon = INITIAL_EPSILON
        while True:

            #神经网络输出的是Q(S,:)值
            action_t = predict_action.eval(feed_dict = {input_image : [input_image_data]})[0]
            argmax_t = np.zeros([output], dtype=np.int)

            #贪心选取动作
            if(random.random() <= INITIAL_EPSILON):
                maxIndex = random.randrange(output)
            else:
                maxIndex = np.argmax(action_t)

            #将action对应的Q(S,a)最大值提取出来
            argmax_t[maxIndex] = 1

            #贪婪的部分开始不断的增加
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            #将选取的动作带入到环境，观察环境状态S'与回报reward
            reward, image = game.step(list(argmax_t))

            #将得到的图形进行变换用于神经网络的输出
            image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
            ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
            image = np.reshape(image, (80, 100, 1))
            input_image_data1 = np.append(image, input_image_data[:, :, 0:3], axis = 2)

            #将S,a,r,S'记录的大脑中
            D.append((input_image_data, argmax_t, reward, input_image_data1))

            #大脑的记忆是有一定的限度的
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            #如果达到观察期就要进行神经网络训练
            if n > OBSERVE:

                #随机的选取一定记忆的数据进行训练
                minibatch = random.sample(D, BATCH)
                #将里面的每一个记忆的S提取出来
                input_image_data_batch = [d[0] for d in minibatch]
                #将里面的每一个记忆的a提取出来
                argmax_batch = [d[1] for d in minibatch]
                #将里面的每一个记忆回报提取出来
                reward_batch = [d[2] for d in minibatch]
                #将里面的每一个记忆的下一步转台提取出来
                input_image_data1_batch = [d[3] for d in minibatch]

                gt_batch = []
                #利用已经有的求解Q(S',:)
                out_batch = predict_action.eval(feed_dict = {input_image : input_image_data1_batch})

                #利用bellman优化得到长期的回报r + γmax(Q(s',:))
                for i in range(0, len(minibatch)):
                    gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

                #利用事先定义的优化函数进行优化神经网络参数
                print("gt_batch:", gt_batch, "argmax:", argmax_batch)
                optimizer.run(feed_dict = {gt : gt_batch, argmax : argmax_batch, input_image : input_image_data_batch})

            input_image_data = input_image_data1
            n = n+1
            print(n, "epsilon:", epsilon, " " ,"action:", maxIndex, " " ,"_reward:", reward)

train_neural_network(input_image)