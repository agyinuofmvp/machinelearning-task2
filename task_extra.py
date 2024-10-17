import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# TODO:解释参数含义，在?处填入合适的参数
batch_size = 64 # 每个批次的样本数量
learning_rate = 0.001 # 学习率
num_epochs = 30
# patience = 5  # 早停的耐心值,即如果验证集损失不减少，训练会停止

# 定义数据预处理管道，将数据转换为PyTorch需要的格式
transform_train = transforms.Compose([
    # 随机裁剪并填充,32指定了裁剪后图像的目标大小,实现过程：每一边添加4像素后再随机选择32 x 32像素区域裁剪
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集，train=True指示加载训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# 创建训练数据加载器，shuffle=True在每个epoch开始时随机打乱数据的顺序，有助于提高模型的泛化能力
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # TODO:这里补全你的网络层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # TODO:这里补全你的前向传播
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # 将四维张量转换为二维张量
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)
        return x
# TODO:补全
model = Network().to(device)

criterion = nn.CrossEntropyLoss()   # 交叉熵损失，适合多类别分类任务
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Adam优化器
# 学习率调度器
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# Early Stopping参数
# best_loss = float('inf') #先设置为无穷大，这样遇到的第一个实际损失值便可替换
# trigger_times = 0  # 验证损失集连续多少次没有改善
#验证函数
'''
def validate():
    model.eval()  #评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    val_loss = val_loss / len(testloader)
    return val_loss, accuracy
'''
# 初始化列表用于记录损失和准确率
train_loss = []
train_acc = []

def train():
    global best_loss, trigger_times
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
        train_loss.append(running_loss / len(trainloader))  # 记录损失
        train_acc.append(accuracy)  # 记录准确率
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')
        # 在每个 epoch 结束后进行验证
        '''
        val_loss, val_accuracy = validate()
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Early Stopping 检查
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0  # 重置耐心计数器
            # 这里可以选择保存模型
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            print(f'Early stopping trigger times: {trigger_times}')

            # 当触发次数超过设定的耐心值时，停止训练
            if trigger_times >= patience:
                print('Early stopping!')
                return
                '''
         # 调整学习率
        # scheduler.step()


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
    # 可视化训练损失和准确率的变化
    plt.figure(figsize=(12, 4)) #新建一个图形窗口，适用于要画多个图形

    # 损失曲线
    plt.subplot(1, 2, 1)  # 1行2列的布局
    plt.plot(train_loss, label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Training Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # 显示图像
    plt.show()
    plt.savefig("task_extra.png")