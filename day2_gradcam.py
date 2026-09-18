import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ---------- 1. 设备与模型（加载你训练好的 LS 模型） ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 37)
model.load_state_dict(torch.load("ls_best.pth", map_location=device))
model = model.to(device).eval()

# ---------- 2. 测试集（划分和训练一模一样） ----------
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
full_dataset = datasets.OxfordIIITPet(root="./data", download=True, transform=None)
labels = [s[1] for s in full_dataset]
train_idx, temp_idx = train_test_split(
    range(len(full_dataset)), test_size=0.3, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=42)
test_set = Subset(datasets.OxfordIIITPet(root="./data", transform=val_transform), test_idx)

ds = datasets.OxfordIIITPet(root="./data", download=False)
class_names = list(ds.classes)

# ---------- 3. 自动挑一张预测正确、一张预测错误的图 ----------
correct_img = wrong_img = None
with torch.no_grad():
    for i in range(len(test_set)):
        img_t, label = test_set[i]
        logit = model(img_t.unsqueeze(0).to(device))
        pred = int(torch.argmax(logit, dim=1).item())
        if correct_img is None and pred == label:
            correct_img = (i, img_t, label, pred)
        if wrong_img is None and pred != label:
            wrong_img = (i, img_t, label, pred)
        if correct_img and wrong_img:
            break
print("正确样本:", class_names[correct_img[2]],
      " 错误样本: 真", class_names[wrong_img[2]], "-> 预测", class_names[wrong_img[3]])

# ---------- 4. Grad-CAM 核心函数 ----------
def gradcam(model, img_t, target_class):
    """对"target_class 这个类别"计算热力图，返回 7x7 的 0~1 矩阵"""
    model.zero_grad()
    act, grad = {}, {}
    def fwd_hook(m, i, o):
        act["v"] = o
    def bwd_hook(m, gi, go):
        grad["v"] = go[0]
    h1 = model.layer4[-1].register_forward_hook(fwd_hook)   # 抓卷积输出
    h2 = model.layer4[-1].register_full_backward_hook(bwd_hook)  # 抓梯度
    logit = model(img_t.unsqueeze(0).to(device))[0, target_class]
    logit.backward()
    h1.remove()
    h2.remove()
    a = act["v"][0]   # [512, 7, 7] 特征图
    g = grad["v"][0]  # [512, 7, 7] 梯度
    w = g.mean(dim=(1, 2), keepdim=True)          # 每个通道的重要程度
    cam = torch.relu((w * a).sum(dim=0)).detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# ---------- 5. 画图：上排正确、下排错误；左原图、右热力图 ----------
def to_rgb(img_t):
    """把归一化张量还原成能显示的图片"""
    img = img_t.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
for row, (title, item) in enumerate([("预测正确", correct_img), ("预测错误", wrong_img)]):
    _, img_t, label, pred = item
    cam = gradcam(model, img_t, pred)   # 对"预测的类别"做热力图
    # 7x7 -> 放大到 224x224 再叠加
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)
    cam_img = np.array(cam_img) / 255.0
    heat = plt.cm.jet(cam_img)[:, :, :3]
    overlay = 0.5 * to_rgb(img_t) + 0.5 * heat

    axes[row, 0].imshow(to_rgb(img_t))
    axes[row, 0].set_title(f"{title}\nTrue: {class_names[label]}", fontsize=10)
    axes[row, 1].imshow(overlay)
    axes[row, 1].set_title(f"Grad-CAM\nPred: {class_names[pred]}", fontsize=10)
    for ax in axes[row]:
        ax.axis("off")

plt.savefig("gradcam_ls.png")
print("已保存: gradcam_ls.png")
