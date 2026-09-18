import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. 设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ---------- 2. 测试集 Transform（和训练时一致，绝不能加随机增强！） ----------
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------- 3. 测试集划分（random_state=42，和训练时一模一样，对比才公平） ----------
full_dataset = datasets.OxfordIIITPet(root="./data", download=True, transform=None)
labels = [s[1] for s in full_dataset]
train_idx, temp_idx = train_test_split(
    range(len(full_dataset)), test_size=0.3, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)

test_set = Subset(datasets.OxfordIIITPet(root="./data", transform=val_transform), test_idx)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
print(f"测试集样本数: {len(test_set)}")

# ---------- 4. 加载训练好的模型 ----------
model = torchvision.models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 37)
model = model.to(device)
model.load_state_dict(torch.load("ls_best.pth", map_location=device))
model.eval()

# ---------- 5. 在测试集上做预测 ----------
all_labels = []
all_preds = []
top1_correct = 0
total = 0
with torch.no_grad():
    for images, batch_labels in test_loader:
        images = images.to(device)
        pred = model(images)                     # [32, 37] 每张图37类的得分
        pred_class = torch.argmax(pred, dim=1)   # 取得分最高的类
        top1_correct += (pred_class.cpu() == batch_labels).sum().item()
        total += batch_labels.size(0)
        all_labels.extend(batch_labels.tolist())
        all_preds.extend(pred_class.cpu().tolist())

# ---------- 6. 算指标 ----------
top1_acc = top1_correct / total
macro_f1 = f1_score(all_labels, all_preds, average="macro")
print(f"测试集 Top-1 准确率: {top1_acc*100:.2f}%")
print(f"测试集 Macro-F1: {macro_f1:.4f}")

# ---------- 7. 画混淆矩阵 + 找最易混淆的类别对 ----------
ds = datasets.OxfordIIITPet(root="./data", download=False)
class_names = list(ds.classes)   # 37 个类名

cm = confusion_matrix(all_labels, all_preds, labels=list(range(37)))
plt.figure(figsize=(13, 11), dpi=150)
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(37), class_names, rotation=90, fontsize=6)
plt.yticks(range(37), class_names, fontsize=6)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)
plt.title(f"Confusion Matrix - Label Smoothing (Top-1: {top1_acc*100:.2f}%)", fontsize=13)
plt.savefig("confusion_ls.png")
print("混淆矩阵已保存: confusion_ls.png")

# 找非对角线数值最大的类别对 = 最容易搞混的品种
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
print("\n最容易混淆的类别对:")
for _ in range(5):
    idx = np.unravel_index(np.argmax(cm_off), cm_off.shape)
    if cm_off[idx] == 0:
        break
    print(f"  {class_names[idx[0]]}  <->  {class_names[idx[1]]}: {int(cm_off[idx])} 张")
    cm_off[idx] = 0
    cm_off[idx[1], idx[0]] = 0
