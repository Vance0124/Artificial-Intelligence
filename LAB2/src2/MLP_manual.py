import torch
from torch import nn
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.autograd import Variable

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 这两行需要手动设置


def DrawDiagram(x, y, x2, y2, path):
    # plt.scatter(x, y, s=10, c='blue')  # 将每个规模和对应的运行时间的对数的散点在图中描出来
    # for a, b in zip(x, y):
    #     plt.text(a, b, (a, b), ha='right', va='bottom', fontsize=10, color='r', alpha=0.5)  # 给这些散点打上标记
    plt.plot(x, y, c='blue', label='manual')  # 描绘出光滑曲线
    plt.plot(x2, y2, c='red', label='torch')  # 描绘出光滑曲线
    plt.legend(loc=1)  # 指定legend图例的位置为右下角
    plt.title("loss损失函数", fontsize=18)  # 标题及字号
    plt.xlabel("迭代次数 n", fontsize=15)  # X轴标题及字号
    plt.ylabel("loss", fontsize=15)  # Y轴标题及字号
    plt.tick_params(axis='both', labelsize=14)  # 刻度大小
    plt.xticks(np.arange(0, 201, 20))
    plt.yticks(np.arange(1.0, 2.1, 0.1))
    plt.savefig(path)
    plt.show()


class MLP_manual:

    def __init__(self, sizes, weights, biases, epochs=200):
        """
            初始化神经网络，给每层的权重和偏置赋初值
            权重为一个列表，列表中每个值是一个二维n×m的numpy数组
            偏置为一个列表，列表中每个值是一个二维n×1的numpy数组
        """
        self.num_layers = len(sizes)  # 神经网络层数
        # 构造神经网络权值矩阵
        self.weights = weights
        # 从第一层隐含层开始添加 偏置项
        self.biases = biases
        self.epochs = epochs

    """
        定义激活函数 sigmoid
    """

    def sigmoid(self, x):
        # print("x:", x)
        y = 1.0 / (1.0 + np.exp(-x))
        # print("y:", y)
        return y

    """
        激活函数 sigmoid 的导数
    """

    def sigmoid_back(self, x):
        # y = np.exp(-x) / (1.0 + np.exp(-x)) ** 2
        y = self.sigmoid(x) * (1 - self.sigmoid(x))
        return y

    """
        第3层的激活函数softmax
    """

    def softmax(self, X):       # 横向的， [3, 100] 这种格式
        # 输入X向量，输出Y向量
        # print(X.shape)
        X_T = X.transpose()
        Y = np.zeros((len(X_T),len(X_T[0])))
        for k in range(len(X_T)):
            c = 0.0
            for i in range(len(X_T[0])):
                c += np.exp(X_T[k][i])
                Y[k][i] = np.exp(X_T[k][i])
            Y[k] = 1 / c * Y[k]
        return Y.transpose()

    """
        交叉熵CrossEntropy
    """

    def CrossEntropy(self, y, y_p):
        y_pred = y_p.transpose()
        loss = np.zeros(len(y))
        sum = 0.0
        for i in range(len(y)):
            loss[i] = -math.log(y_pred[i][y[i] - 1])  # 按照交叉熵定义
            sum += abs(loss[i])
        sum = sum / len(y)
        return sum, loss

    """
        交叉熵导数l' * s3'
    """

    def nabla_ls(self, y, y_p):
        """
        :param y: 输入结果，监督学习，一维,类别从 1 开始
        :param y_pred: 预测结果，2维（第2维3个元素），注意，y_pred的下标从 0 开始
        :return: 导数乘积
        """
        y_pred = y_p.transpose()
        ans = np.zeros((len(y_pred), len(y_pred[0])))
        for k in range(len(y)):
            for i in range(len(y_pred[0])):
                if i + 1 == y[k]:
                    ans[k][i] = y_pred[k][i] - 1
                else:
                    ans[k][i] = y_pred[k][i]
        return ans.transpose()

    """
        前向传播 feed_forward
    """

    def feed_forward(self, X):
        # 前向传播
        vec = X.transpose()
        for i in range(len(self.weights) - 1):
            vec = self.sigmoid(np.dot(self.weights[i], vec) + self.biases[i])
        # 最后一层
        vec = self.softmax(np.dot(self.weights[len(self.weights) - 1], vec) + self.biases[len(self.weights) - 1])
        Y = vec  # 最终结果
        return Y

    """
        反向传播函数 feed_back
    """

    def feed_back(self, x, y):
        # y : 输入的结果标签，监督学习
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # 计算导数，反向传播
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # 前向传播，计算各层的激活前的输出值以及激活之后的输出值，为下一步反向传播计算作准备
        activations = [x.transpose()]
        zs = []
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            zs.append(z)
            activation = self.sigmoid(z)  # 激活函数
            activations.append(activation)
        z = np.dot(self.weights[len(self.weights) - 1], activations[-1]) + self.biases[len(self.weights) - 1]
        zs.append(z)
        activation = self.softmax(z)  # softmax
        activations.append(activation)

        # 先求最后一层的delta误差以及b和W的导数
        delta = self.nabla_ls(y, activations[-1])
        lossavg, loss = self.CrossEntropy(y, activations[-1])  # 损失函数值
        nabla_b[-1] = np.sum(delta, axis=1)/len(delta[0])    # 按行求平均
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 将delta误差反向传播以及各层b和W的导数，一直计算到第二层
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.sigmoid_back(zs[-l])
            nabla_b[-l] = np.sum(delta,axis=1)/len(delta[0])   # 按行求平均
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w, lossavg

    """
        梯度下降
    """

    def gradient_descent(self, x, y, lr=0.001):
        """
        :param x:   输入
        :param y:   输出
        :param lr:  学习速率
        :return:    权值矩阵，训练好的
        """
        X = np.zeros(self.epochs)
        loss = np.zeros(self.epochs)
        for i in range(self.epochs):
            DNB, DNW, loss[i] = self.feed_back(x, y)
            print("权值W的梯度： ", DNW)
            print("偏置项b的梯度： ", DNB)
            X[i] = i
            self.weights = [w - lr / len(x) * nw for w, nw in zip(self.weights, DNW)]
            self.biases = [b - lr * nb.reshape(b.shape) for b, nb in zip(self.biases, DNB)]
        print("最终权值W： ", self.weights)
        print("偏置项b：", self.biases)
        return X, loss


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 使用父类的初始化参数
        self.mlp = nn.Sequential(
            nn.Linear(5, 4),
            nn.Sigmoid(),
            nn.Linear(4, 4),
            nn.Sigmoid(),
            nn.Linear(4, 3),
            # nn.Softmax(1)  # torch.nn.CrossEntropyLoss() 已有 Softmax层，不需要重复定义
        )
        # 定义神经网络里的输入、隐藏和输出层

    def initial(self, weights, biases):
        i = 0
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):  # 判断是否是线性层
                # print("before: ", layer.weight.data)
                # print(layer.bias.shape, biases[i].shape, type(layer.bias))
                layer.weight.data = torch.from_numpy(weights[i])  # double类型
                layer.bias.data = torch.from_numpy(biases[i])
                # print(layer.bias.shape, type(layer.bias))
                # print(layer.weight.data)
                i = i + 1

    def forward(self, x):
        """
         在forward函数中，我们会接受一个Variable，然后我们也会返回一个Varible
        """
        y_pred = self.mlp(x)
        return y_pred


def main():
    sizes = [5, 4, 4, 3]
    X = np.random.randn(100, 5)
    Y = np.random.randint(1, 4, 100)
    print("X: ", X)
    print("Y: ", Y)
    W = [np.random.randn(n, m) for m, n in zip(sizes[:-1], sizes[1:])]  # 一定得用randn而不是random
    B = [np.random.randn(n, 1) for n in sizes[1:]]
    # 设置超参数
    learning_rate = 0.01
    EPOCH = 200
    path = '../photos/loss.png'
    NET = MLP_manual(sizes=sizes, weights=W, biases=B)
    xc, yc = NET.gradient_descent(X, Y, learning_rate)

    # 生成随机数当作样本，同时用Variable 来包装这些数据，设置 requires_grad=False 表示在方向传播的时候，
    # 我们不需要求这几个 Variable 的导数
    X_torch = Variable(torch.from_numpy(X))
    Y_torch = Variable(torch.from_numpy(Y) - 1).long()

    net_torch = MLP()
    B_torch = [np.random.randn(n) for n in sizes[1:]]   # 匹配格式
    for i in range(len(B)):
        B_torch[i] = B[i].reshape(B_torch[i].shape)
    net_torch.initial(W, B_torch)

    # 定义损失函数
    loss_fn = torch.nn.CrossEntropyLoss()

    # 使用optim包来定义优化算法，可以自动的帮我们对模型的参数进行梯度更新。这里我们使用的是随机梯度下降法。
    # 第一个传入的参数是告诉优化器，我们需要进行梯度更新的Variable 是哪些，
    # 第二个参数就是学习速率了。
    optimizer = torch.optim.SGD(net_torch.parameters(), lr=learning_rate)

    # 开始训练
    xc_t = xc
    yc_t = np.zeros(EPOCH)
    for t in range(EPOCH):
        # 向前传播
        y_pred = net_torch.forward(X_torch)
        # 计算损失
        loss = loss_fn(y_pred, Y_torch)
        # 显示损失
        yc_t[t] = loss
        # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
        optimizer.zero_grad()
        # 计算梯度
        loss.backward()
        # 更新梯度
        optimizer.step()
    DrawDiagram(xc, yc, xc_t, yc_t, path)   # 画图


main()
