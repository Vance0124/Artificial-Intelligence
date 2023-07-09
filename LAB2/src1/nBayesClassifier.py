import numpy as np
from process_data import load_and_process_data
from evaluation import get_micro_F1,get_macro_F1,get_acc

class NaiveBayes:
    '''参数初始化
    Pc: P(c) 每个类别c的概率分布
    Pxc: P(c|x) 每个特征的条件概率
    '''
    def __init__(self):
        self.Pc={}
        self.Pxc={}

    '''
    采用第二种方法，计算高斯概率密度函数
    mean: 平均值
    std: 方差
    '''
    def GaussProb(self, x, mean, std):
        exponent = np.exp(-(np.power(x - mean, 2)) / (2 * np.power(std, 2)))
        GaussProb = (1 / (np.sqrt(2 * np.pi) * std)) * exponent
        return GaussProb

    '''
    通过训练集计算先验概率分布p(c)和条件概率分布p(x|c)
    建议全部取log，避免相乘为0
    '''
    def fit(self,traindata,trainlabel,featuretype):
        '''
        需要你实现的部分
        '''
        # 先计算先验概率和离散的 Sex 的后验概率
        Label = {}  # 每个标签的数量
        Sex = np.zeros((3, 3))  # 3 * 3 矩阵，表示3个类别3种性别的数量
        totalnum = len(traindata)
        for l in range(len(trainlabel)):
            if trainlabel[l][0] in Label:
                Label[trainlabel[l][0]] += 1
            else:
                Label[trainlabel[l][0]] = 1
            Sex[trainlabel[l][0]-1][int(traindata[l][0])-1] += 1     # 每个类别中每种性别的数量
        for key, value in Label.items():
            self.Pc[key] = (value + 1) / (totalnum + 3)     # N = 3

        SexSum = list(map(sum, Sex))    # 求出每一个类别的总数
        self.Pxc["class1"] = {}
        for i in range(3):
            for j in range(3):
                self.Pxc["class1"]["%d-%d" % (i+1, j+1)] = (Sex[i][j] + 1) / (SexSum[i] + 3)
        # print(self.Pxc)

        # 开始处理连续变量
        for i in range(1, 8):
            par = {}
            Classes = [[], [], []]

            for j in range(len(traindata)):     # 分类不同类别的属性
                Classes[trainlabel[j][0] - 1].append(traindata[j][i])

            par["1"] = [np.mean(np.array(Classes[0])), np.std(np.array(Classes[0]))]    # 计算参数
            par["2"] = [np.mean(np.array(Classes[1])), np.std(np.array(Classes[1]))]
            par["3"] = [np.mean(np.array(Classes[2])), np.std(np.array(Classes[2]))]
            self.Pxc["class%d" % (i+1)] = par

        # print(self.Pxc)

    '''
    根据先验概率分布p(c)和条件概率分布p(x|c)对新样本进行预测
    返回预测结果,预测结果的数据类型应为np数组，shape=(test_num,1) test_num为测试数据的数目
    feature_type为0-1数组，表示特征的数据类型，0表示离散型，1表示连续型
    '''
    def predict(self,features,featuretype):
        '''
        需要你实现的部分
        '''       
        pred = []
        S = [1, 1, 1]
        for i in range(len(features)):
            S[0] = self.Pc[1] * self.Pxc["class1"]["%d-%d" % (1, features[i][0])]      # 判断这3类哪一类概率最高
            S[1] = self.Pc[2] * self.Pxc["class1"]["%d-%d" % (2, features[i][0])]
            S[2] = self.Pc[3] * self.Pxc["class1"]["%d-%d" % (3, features[i][0])]
            for j in range(1, 8):
                S[0] = S[0] * self.GaussProb(features[i][j], self.Pxc["class%d" % (j+1)]["1"][0], self.Pxc["class%d" % (j+1)]["1"][1])
                S[1] = S[1] * self.GaussProb(features[i][j], self.Pxc["class%d" % (j+1)]["2"][0], self.Pxc["class%d" % (j+1)]["2"][1])
                S[2] = S[2] * self.GaussProb(features[i][j], self.Pxc["class%d" % (j+1)]["3"][0], self.Pxc["class%d" % (j+1)]["3"][1])
            pred.append(S.index(max(S))+1)  # 预测类别
        pred = np.array(pred).astype(int).reshape(-1, 1)
        return pred


def main():
    # 加载训练集和测试集
    train_data,train_label,test_data,test_label=load_and_process_data()
    feature_type=[0,1,1,1,1,1,1,1] #表示特征的数据类型，0表示离散型，1表示连续型

    Nayes=NaiveBayes()
    Nayes.fit(train_data,train_label,feature_type) # 在训练集上计算先验概率和条件概率

    pred=Nayes.predict(test_data,feature_type)  # 得到测试集上的预测结果

    # 计算准确率Acc及多分类的F1-score
    print("Acc: "+str(get_acc(test_label,pred)))
    print("macro-F1: "+str(get_macro_F1(test_label,pred)))
    print("micro-F1: "+str(get_micro_F1(test_label,pred)))

main()