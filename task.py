import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO:解释参数含义，在?处填入合适的参数
batch_size = 64 # 每个批次的样本数量
learning_rate = 0.001 # 学习率
num_epochs = 10

# 定义数据预处理管道，将数据转换为PyTorch需要的格式
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载CIFAR-10数据集，train=True指示加载训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# 创建训练数据加载器，shuffle=True在每个epoch开始时随机打乱数据的顺序，有助于提高模型的泛化能力
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # TODO:这里补全你的前向传播
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 将四维张量转换为二维张量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# TODO:补全
model = Network().to(device)

criterion = nn.CrossEntropyLoss()   # 交叉熵损失，适合多类别分类任务
optimizer = optim.Adam(model.parameters(),lr=learning_rate)  # Adam优化器

def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 提取预测的类别，1指定了维度，表示在每一行寻找最大值及其对应的索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

if __name__ == "__main__":
    train()
    test()