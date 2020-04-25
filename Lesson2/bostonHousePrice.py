"""
    使用numpy实现Boston房价预测
    Step1 数据加载，来源sklearn中的load_boston
    Step2 数据规范化，将X 采用正态分布规范化
    Step3 初始化网络
    Step4 定义激活函数，损失函数，学习率 epoch
    Step5 循环执行：前向传播，计算损失函数，反向传播，参数更新
    Step6 输出训练好的model参数，即w1, w2, b1, b2
"""
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt

def init_network(input,output,d_hidden):
    '''
    网络初始化
    :param input: x
    :param output: y
    :param d_hidden: 隐藏层维度
    :return: w1,b1,w2,b2
    '''
    (d_in,n) =input.shape
    d_out = output.shape[1]

    w1 = np.random.randn(n,d_hidden)
    b1 = np.random.randn(d_in,d_hidden)

    w2 = np.random.randn(d_hidden,d_out)
    b2 = np.random.randn(d_in,d_out)
    return [w1,b1,w2,b2]

def relu(x):
    '''
    relu激活函数
    '''
    return np.where(x>0,x,0)

def linear_func(x,w,b):
    '''
    线性回归模型
    :param x: 输入
    :return: 预测值
    '''
    y_hat = np.dot(x,w) + b
    return y_hat

def forward(x,w1,b1,w2,b2):
    '''
    前向传播
    '''
    temp = linear_func(x,w1,b1)
    temp_relu = relu(temp)
    y_pred = linear_func(temp,w2,b2)
    return temp,temp_relu,y_pred

def MSE_loss(y,y_pred):
    '''
    损失函数
    '''
    loss = np.square(y-y_pred).sum()
    return loss

# 数据采集
data = load_boston()
x = data['data']
y = data['target']
y = y.reshape(y.shape[0],1)
#print(y.shape)

# 数据标准化
x = (x-np.mean(x,axis=0))/np.std(x)
#print(x.shape)

# 网络初始化
[W1,b1,W2,b2] = init_network(x,y,d_hidden = 10)
#print(W1.shape,b1.shape,W2.shape,b2.shape)

# 设置学习率
learning_rate = 1e-6

# 设置迭代次数 i=5000
for i in range(5000):
    # 前向传播
    temp,temp_relu,y_pred = forward(x,W1,b1,W2,b2)

    # 计算损失函数
    loss = MSE_loss(y,y_pred)
    print('w1 = {} w2 = {}: loss = {}'.format(W1,W2,loss))

    # 反向传播
    grad_y_pred = 2.0 * (y_pred - y)
    # print('grad_y_pred=', grad_y_pred.shape) #(64, 10)
    grad_w2 = temp_relu.T.dot(grad_y_pred)
    grad_temp_relu = grad_y_pred.dot(W2.T)
    grad_temp = grad_temp_relu.copy()
    grad_temp_relu[temp<0] = 0
    grad_w1 = x.T.dot(grad_temp)

    # 更新权重
    W1 -= learning_rate * grad_w1
    W2 -= learning_rate * grad_w2
print('===============================')
print('w1={} \n w2={}'.format(W1, W2))




