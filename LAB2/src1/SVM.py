import numpy as np
import cvxopt #用于求解线性规划
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc


#根据指定类别main_class生成1/-1标签
def svm_label(labels,main_class):
    new_label=[]
    for i in range(len(labels)):
        if labels[i]==main_class:
            new_label.append(1)
        else:
            new_label.append(-1)
    return np.array(new_label)

# 实现线性回归
class SupportVectorMachine:

    '''参数初始化 
    lr: 梯度更新的学习率
    Lambda: L2范数的系数
    epochs: 更新迭代的次数
    '''
    def __init__(self,kernel,C,Epsilon):
        self.kernel=kernel
        self.C = C
        self.Epsilon=Epsilon

    '''KERNEL用于计算两个样本x1,x2的核函数'''
    def KERNEL(self, x1, x2, kernel='Gauss', d=2, sigma=1):
        #d是多项式核的次数,sigma为Gauss核的参数
        K = 0
        if kernel == 'Gauss':
            K = np.exp(-(np.sum((x1 - x2) ** 2)) / (2 * sigma ** 2))
        elif kernel == 'Linear':
            K = np.dot(x1,x2)
        elif kernel == 'Poly':
            K = np.dot(x1,x2) ** d
        else:
            print('No support for this kernel')
        return K

    '''
    根据训练数据train_data,train_label（均为np数组）求解svm,并对test_data进行预测,返回预测分数，即svm使用符号函数sign之前的值
    train_data的shape=(train_num,train_dim),train_label的shape=(train_num,) train_num为训练数据的数目，train_dim为样本维度
    预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    '''
    def fit(self,train_data,train_label,test_data):
        '''
        需要你实现的部分
        '''
        train_num = len(train_label)
        # 先求 X 的核矩阵
        K = np.zeros((train_num, train_num))
        for i in range(train_num):
            for j in range(train_num):
                K[i][j] = self.KERNEL(train_data[i], train_data[j], kernel=self.kernel)
        P = cvxopt.matrix(np.outer(train_label, train_label) * K)   # 标签已经 (-1,1)化
        # P = np.zeros((train_num, train_num))
        q = cvxopt.matrix(np.ones(train_num) * -1)   #为列向量
        A = cvxopt.matrix(train_label, (1, train_num), 'd')    #y的转置为行向量（1, train_num）表示排列为1* train_num 的矩阵
        b = cvxopt.matrix(0.0)
        # print(A, len(A))
        # os.system("pause")
        #对于线性可分数据集
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(train_num) * -1))
            h = cvxopt.matrix(np.zeros(train_num))
        #对于软间隔
        else:
            arg1 = np.diag(np.ones(train_num) * -1)
            arg2 = np.diag(np.ones(train_num))
            G = cvxopt.matrix(np.vstack((arg1,arg2)))   #加括号  因为vstack只需要一个参数，纵向堆叠
            arg1 = np.zeros(train_num)
            arg2 = np.ones(train_num) * self.C
            h = cvxopt.matrix(np.hstack((arg1,arg2)))   # 横向堆叠

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        sol = np.ravel(solution['x'])   # 变成一个一维数组, alpha 的解

        if self.C is None:
            sv = sol > self.Epsilon
        else:
            sv = (sol > self.Epsilon) * (sol < self.C)

        index = np.arange(len(sol))[sv]   #sol > 1e-15  返回true or false的数组   取出对应为 true 的下标

        self.alpha = sol[sv]  # 大于0对应的sol的值 index
        self.sv_X = train_data[sv]  # 支持向量的X index
        self.sv_y = train_label[sv]    # 支持向量的y  index

        # print('%d 点中有 %d 个支持向量' % ( train_num,len(self.alpha))) # 按照定义

        # 算 w 以及 b
        # 求 w
        self.w = np.zeros(len(train_data[0]))    # 总共 8 个特征
        for i in range(len(self.alpha)):
            self.w += self.alpha[i] * self.sv_y[i] * self.sv_X[i]
        print("W: ", self.w)
        # 求b
        self.b = 0
        for i in range(len(self.alpha)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.alpha * self.sv_y * K[index[i]][index])
            # self.b -= np.sum(self.alpha * self.sv_y * P[index[i]][index])
        self.b /= len(self.alpha)   # 取平均
        print("b:", self.b)

        pred = np.dot(test_data, self.w) + self.b
        # print("pred: ", pred)
        return pred



def main():
    # 加载训练集和测试集
    Train_data,Train_label,Test_data,Test_label=load_and_process_data()
    Train_label=[label[0] for label in Train_label]
    Test_label=[label[0] for label in Test_label]
    train_data=np.array(Train_data)
    test_data=np.array(Test_data)
    test_label=np.array(Test_label).reshape(-1,1)
    #类别个数
    num_class=len(set(Train_label))


    #kernel为核函数类型，可能的类型有'Linear'/'Poly'/'Gauss'
    #C为软间隔参数；
    #Epsilon为拉格朗日乘子阈值，低于此阈值时将该乘子设置为0
    # kernel='Linear'
    # kernel = 'Gauss'
    kernel = 'Poly'
    C = 1
    Epsilon=10e-5
    #生成SVM分类器
    SVM=SupportVectorMachine(kernel,C,Epsilon)

    predictions = []
    #one-vs-all方法训练num_class个二分类器
    for k in range(1,num_class+1):
        #将第k类样本label置为1，其余类别置为-1
        train_label=svm_label(Train_label,k)
        # 训练模型，并得到测试集上的预测结果
        prediction=SVM.fit(train_data,train_label,test_data)
        predictions.append(prediction)
    predictions=np.array(predictions)
    print(predictions)
    #one-vs-all, 最终分类结果选择最大score对应的类别
    pred = np.argmax(predictions,axis=0)+1
    pred = np.array(pred).astype(int).reshape(-1, 1)

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))


main()
