import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
#禁止import除了torch以外的其他包，依赖这几个包已经可以完成实验了

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mixer_Layer(nn.Module):
    def __init__(self, patch_size, hidden_dim):
        super(Mixer_Layer, self).__init__()
        ########################################################################
        #这里需要写Mixer_Layer（layernorm，mlp1，mlp2，skip_connection）
        # self.patch_size = patch_size
        # self.hidden_dim = hidden_dim
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.mlp1 = nn.Sequential(  # 第一个全连接层
            nn.Linear((28 // patch_size) ** 2, 256),
            nn.GELU(),
            nn.Linear(256, (28 // patch_size) ** 2)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.mlp2 = nn.Sequential(  # 第2个全连接层
            nn.Linear(hidden_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, hidden_dim)
        )
        ########################################################################

    def forward(self, x):
        ########################################################################
        y = self.layer_norm1(x) # layernorm
        y = y.transpose(1, 2)   # 转置
        y = self.mlp1(y)    # 第一个全连接层
        y = y.transpose(1, 2)       # 转置回来
        x = x + y      # skip_connection

        y = self.layer_norm2(x)
        y = self.mlp2(y)
        return x + y
        ########################################################################


class MLPMixer(nn.Module):
    def __init__(self, patch_size, hidden_dim, depth):
        super(MLPMixer, self).__init__()
        assert 28 % patch_size == 0, 'image_size must be divisible by patch_size'
        assert depth > 1, 'depth must be larger than 1'
        ########################################################################
        #这里写Pre-patch Fully-connected, Global average pooling, fully connected

        #  Per-patch Fully-connected 相当于 embedding ( 嵌入 ) 层
        self.embedding = nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        mix_layer = Mixer_Layer(patch_size, hidden_dim)
        self.mixer_layers = nn.Sequential(*[mix_layer for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)    # 归一化
        self.cls = nn.Linear(hidden_dim, 10)    # 最后一个全连接层预测类别， 共 10类
        ########################################################################


    def forward(self, data):
        ########################################################################
        #注意维度的变化
        emb_out = self.embedding(data).flatten(2)
        emb_out = emb_out.transpose(1, 2)
        # 从第dim个维度开始展开，将后面的维度转化为一维，只保留dim之前的维度，其他维度的数据全都挤在dim这一维
        mix_out = self.mixer_layers(emb_out)
        norm_out = self.norm(mix_out)
        data_avg = torch.mean(norm_out, dim=1)  # 逐通道求均值 Global Average Pooling
        C = self.cls(data_avg)
        return C
        ########################################################################


def train(model, train_loader, optimizer, n_epochs, criterion):
    model.train()
    for epoch in range(n_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ########################################################################
            #计算loss并进行优化
            output = model(data)
            loss = criterion(output, target)
            # 在我们进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度。
            optimizer.zero_grad()
            # 计算梯度
            loss.backward()
            # 更新梯度
            optimizer.step()
            ########################################################################
            if batch_idx % 100 == 0:
                print('Train Epoch: {}/{} [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, n_epochs, batch_idx * len(data), len(train_loader.dataset), loss.item()))


def test(model, test_loader, criterion):
    # print("test: ", test_loader)
    model.eval()
    test_loss = 0.
    num_correct = 0 #correct的个数
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
        ########################################################################
        #需要计算测试集的loss和accuracy
            out = model(data)
            # print("out: ", out)
            # print("target ", target.size(0), len(target))
            # print(target)
            loss = criterion(out, target)
            value_max, pred = torch.max(out, 1)     # 按行索引，得到每行最大值及其索引
            test_loss += loss.data * len(target)    # len(target) : 预测图片数量
            for i in range(len(target)):
                if pred[i] == target[i]:
                    num_correct += 1

        totalnum = 0     # 统计预测的总数量
        for _, target in test_loader:
            target = target.to(device)
            totalnum += len(target)
            # accuracy = torch.sum(target == out) / len(target)
        print("total: ", totalnum)
        test_loss = test_loss / totalnum
        accuracy = num_correct / totalnum
        ########################################################################
        print("Test set: Average loss: {:.4f}\t Acc {:.2f}".format(test_loss.item(), accuracy))




if __name__ == '__main__':
    n_epochs = 5
    batch_size = 128
    learning_rate = 1e-3

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MNIST(root = './data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = MNIST(root = './data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    ########################################################################
    model = MLPMixer(patch_size=4, hidden_dim=256, depth=8).to(device) # 参数自己设定，其中depth必须大于1
    # 这里需要调用optimizer，criterion(交叉熵)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    ########################################################################
    
    train(model, train_loader, optimizer, n_epochs, criterion)
    test(model, test_loader, criterion)