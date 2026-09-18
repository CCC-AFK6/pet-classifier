import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 1. 读取两个实验的 TensorBoard 日志 ----------
runs = {"Baseline": "runs/pet_resnet18", "Label Smoothing": "runs/pet_ls"}
data = {}
for name, path in runs.items():
    ea = EventAccumulator(path)
    ea.Reload()
    data[name] = {}
    for tag in ea.Tags()["scalars"]:   # 里面有 Loss/Train、Loss/Val、Acc/Val
        data[name][tag] = [e.value for e in ea.Scalars(tag)]

# ---------- 2. 画图：左图 Loss 曲线，右图验证准确率曲线 ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)
colors = {"Baseline": "#1f77b4", "Label Smoothing": "#ff7f0e"}

for name, d in data.items():
    c = colors[name]
    ep = list(range(1, len(d["Loss/Train"]) + 1))   # epoch 1~10
    axes[0].plot(ep, d["Loss/Train"], color=c, linestyle="-", label=f"{name} (train)")
    axes[0].plot(ep, d["Loss/Val"], color=c, linestyle="--", label=f"{name} (val)")
    axes[1].plot(ep, d["Acc/Val"], color=c, marker="o", label=name)

axes[0].set_title("Train / Val Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].set_title("Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Val Acc")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.savefig("training_curves.png")
print("已保存: training_curves.png")
