import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler  # 引入缩放的包


# 线性代价函数
def compute_cost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 梯度下降算法
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)

    temp = np.matrix(np.zeros((n, num_iters)))  # 暂存每次迭代计算的theta，转化为矩阵形式

    J_history = np.zeros((num_iters, 1))  # 记录每次迭代计算的代价值

    for i in range(num_iters):  # 遍历迭代次数
        h = np.dot(X, theta)  # 计算内积，matrix可以直接乘
        temp[:, i] = theta - ((alpha / m) * (np.dot(np.transpose(X), h - y)))  # 梯度的计算
        theta = temp[:, i]
        J_history[i] = compute_cost(X, y, theta)  # 调用计算代价函数
        print
        '.',
    return theta, J_history


# 归一化feature
def feature_normaliza(X):
    X_norm = np.array(X)  # 将X转化为numpy数组对象，才可以进行矩阵的运算
    # 定义所需变量
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mu = np.mean(X_norm, 0)  # 求每一列的平均值（0指定为列，1代表行）
    sigma = np.std(X_norm, 0)  # 求每一列的标准差
    for i in range(X.shape[1]):  # 遍历列
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]  # 归一化

    return X_norm, mu, sigma


# 正规方程
def normal_eqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X等价于X.T.dot(X)
    return theta


# 逻辑函数为S形的函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 代价函数
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# 逻辑回归的代价函数
def cost_reg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2)))
    return np.sum(first - second) / (len(X)) + reg


# 逻辑回归正则化代价函数
def cost_function(initial_theta, X, y, inital_lambda):
    m = len(y)
    J = 0

    h = sigmoid(np.dot(X, initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()  # 因为正则化j=1从1开始，不包含0，所以复制一份，前theta(0)值为0
    theta1[0] = 0

    temp = np.dot(np.transpose(theta1), theta1)
    J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y),
                                                      np.log(1 - h)) + temp * inital_lambda / 2) / m  # 正则化的代价方程
    return J


# 计算正则化后的代价的梯度
def gradient(initial_theta, X, y, inital_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))

    h = sigmoid(np.dot(X, initial_theta))  # 计算h(z)
    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(np.transpose(X), h - y) / m + inital_lambda / m * theta1  # 正则化的梯度
    return grad


# 映射为多项式
def map_feature(X1, X2):
    degree = 3;  # 映射的最高次方
    out = np.ones((X1.shape[0], 1))  # 映射后的结果数组（取代X）
    '''
    因为数据的feture可能很少，导致偏差大，所以创造出一些feture结合
eg:映射为2次方的形式:1 + {x_1} + {x_2} + x_1^2 + {x_1}{x_2} + x_2^2
    这里以degree=2为例，映射为1,x1,x2,x1^2,x1,x2,x2^2
    '''
    for i in np.arange(1, degree + 1):
        for j in range(i + 1):
            temp = X1 ** (i - j) * (X2 ** j)  # 矩阵直接乘相当于matlab中的点乘.*
            out = np.hstack((out, temp.reshape(-1, 1)))
    return out
