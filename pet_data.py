import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# ========== 1. 设置设备（优先用你的RTX4060 GPU） ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ========== 2. 定义预处理Transform ==========
# 训练集：带随机增强
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 验证 & 测试集：无随机操作，中心裁剪
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

])

# ========== 3. 加载OxfordIIITPet数据集 ==========
# 先下载原始数据集（不带transform，方便分层划分）
full_dataset = datasets.OxfordIIITPet(root="./data", download=True, transform=None)
labels = [sample[1] for sample in full_dataset]  # 获取全部样本标签，用于分层抽样

# ========== 4. 分层划分 70%训练 /15%验证 /15%测试 ==========
# 第一步：分出70%训练集，剩下30%作为val+test
train_idx, temp_idx = train_test_split(
    range(len(full_dataset)),
    test_size=0.3,
    stratify=labels,  # stratify 就是分层抽样！保证类别比例不变
    random_state=42
)
# 第二步：把剩下30%对半拆成验证15%，测试15%
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=[labels[i] for i in temp_idx],
    random_state=42
)

# 分别绑定对应的transform
train_set = Subset(datasets.OxfordIIITPet(root="./data", transform=train_transform), train_idx)
val_set   = Subset(datasets.OxfordIIITPet(root="./data", transform=val_test_transform), val_idx)
test_set  = Subset(datasets.OxfordIIITPet(root="./data", transform=val_test_transform), test_idx)

print(f"训练集样本数: {len(train_set)}")
print(f"验证集样本数: {len(val_set)}")
print(f"测试集样本数: {len(test_set)}")

from torch.utils.data import DataLoader

# 构建数据加载器
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

# ---------------------- 1. 设备选择（你已经有cuda） ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------------------- 2. 构建ResNet18模型 ----------------------
# 加载带预训练权重的resnet18
model = models.resnet18(weights="DEFAULT")
# resnet18原始最后一层输出是1000类（ImageNet），我们要改成37类
in_features = model.fc.in_features  # 获取最后一层输入维度
model.fc = nn.Linear(in_features, 37) # 替换全连接层，输出37类
model = model.to(device) # 把模型放到GPU上

# ---------------------- 3. 损失函数、优化器 ----------------------
loss_fn = nn.CrossEntropyLoss() # 分类任务标准损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ---------------------- 4. TensorBoard日志记录器 ----------------------
# 日志会存到 runs/ 文件夹，后面可以在终端打开tensorboard看曲线
writer = SummaryWriter(log_dir="runs/pet_resnet18")

# ---------------------- 5. 训练一轮函数 ----------------------
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train() # 开启训练模式（启用dropout、bn更新）
    total_loss = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)         # 模型前向推理，得到预测分数
        loss = loss_fn(pred, labels) # 计算损失

        optimizer.zero_grad() # 清空上一轮梯度
        loss.backward()       # 反向传播，计算梯度
        optimizer.step()      # 更新模型权重

        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss

# ---------------------- 6. 验证一轮函数（不更新权重！） ----------------------
def val_one_epoch(model, loader, loss_fn, device):
    model.eval() # 评估模式，关闭dropout，不更新BN
    total_loss = 0
    correct = 0
    total_samples = 0
    # 验证不需要计算梯度，节省显存
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            total_loss += loss.item()

            # 计算准确率
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / len(loader)
    acc = correct / total_samples
    return avg_loss, acc

# ---------------------- 7. 主训练循环 ----------------------
num_epochs = 10  # 任务要求 10~15 epoch，先用10轮试试
best_acc = 0.0   # 记录目前最好的验证准确率

for epoch in range(num_epochs):
    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = val_one_epoch(model, val_loader, loss_fn, device)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 写入TensorBoard日志
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", val_loss, epoch)
    writer.add_scalar("Acc/Val", val_acc, epoch)

    # 如果当前验证准确率是历史最高，保存模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ 保存最优模型，当前最佳准确率: {best_acc:.4f}")

writer.close()
print("\n训练结束！最优模型已保存 best_model.pth")
