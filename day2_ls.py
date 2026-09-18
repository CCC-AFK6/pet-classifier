import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter   # ★ 新增：TensorBoard

# ---------- 1. 设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 2. Transform（和 Day1 完全一样，别动） ----------
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- 3. 数据集 + 分层划分（和 Day1 完全一样，保证对比公平） ----------
full_dataset = datasets.OxfordIIITPet(root="./data", download=True, transform=None)
labels = [s[1] for s in full_dataset]
train_idx, temp_idx = train_test_split(
    range(len(full_dataset)), test_size=0.3, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)

train_set = Subset(datasets.OxfordIIITPet(root="./data", transform=train_transform), train_idx)
val_set = Subset(datasets.OxfordIIITPet(root="./data", transform=val_test_transform), val_idx)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# ---------- 4. 模型 ----------
model = torchvision.models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 37)
model = model.to(device)

# ★ 消融实验唯一改动：Label Smoothing
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# ★ 新增：TensorBoard 日志记录器（日志存到 runs/pet_ls）
writer = SummaryWriter(log_dir="runs/pet_ls")

# ---------- 5. 训练/验证函数 ----------
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)
            total_loss += loss.item()
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

# ---------- 6. 训练循环 ----------
num_epochs = 10
best_acc = 0.0
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_acc = val_one_epoch(model, val_loader, loss_fn, device)
    print(f"[{epoch+1}/{num_epochs}] Train Loss {train_loss:.4f} | "
          f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

    # ★ 新增：把每轮损失和准确率写进日志
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Val", val_loss, epoch)
    writer.add_scalar("Acc/Val", val_acc, epoch)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "ls_best.pth")
        print(f"  -> 保存最优模型 (Val Acc {best_acc:.4f})")

writer.close()  # ★ 新增：关闭日志
print(f"训练完成！最优验证准确率: {best_acc:.4f}")

