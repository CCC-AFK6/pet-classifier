# 牛津宠物 37 类细粒度分类（ResNet18 消融实验）

基于 Oxford-IIIT Pet 数据集（37 类猫狗品种）的细粒度图像分类项目，对比 **Baseline（ResNet18）** 与 **Label Smoothing（标签平滑，α=0.1）** 两种训练策略的泛化表现。

## 环境依赖

- Python 3.8+
- PyTorch / torchvision
- scikit-learn
- matplotlib
- tensorboard
- Pillow

安装依赖：

```bash
pip install torch torchvision scikit-learn matplotlib tensorboard pillow
```

## 数据集

使用 `torchvision.datasets.OxfordIIITPet` 自动下载（首次运行需联网，数据保存于 `data/`，已通过 `.gitignore` 忽略）：
- 类别数：37 类
- 数据划分：70% 训练 / 15% 验证 / 15% 测试，分层抽样，`random_state=42`（保证两轮实验对比公平）

## 项目结构

| 文件 | 作用 |
|---|---|
| `pet_data.py` | 数据加载与预处理 |
| `day2_ls.py` | Label Smoothing 训练（TensorBoard 日志 → `runs/pet_ls`，最优模型 → `ls_best.pth`） |
| `day2_evaluate.py` | 测试集评估：Top-1 准确率、Macro-F1、混淆矩阵（→ `confusion_ls.png`） |
| `day2_curves.py` | 读取 TensorBoard 日志绘制训练曲线（→ `training_curves.png`） |
| `day2_gradcam.py` | Grad-CAM 可视化（→ `gradcam_ls.png`） |
| `runs/` | 两轮实验的 TensorBoard 训练日志 |

## 一键复现

以下命令在项目根目录执行，脚本自动选择 GPU/CPU：

```bash
# 1. 训练（10 epochs，最优模型保存为 ls_best.pth）
python day2_ls.py

# 2. 测试集评估（输出 Top-1 / Macro-F1，生成混淆矩阵）
python day2_evaluate.py

# 3. 绘制 Baseline vs Label Smoothing 训练曲线
python day2_curves.py

# 4. Grad-CAM 可视化
python day2_gradcam.py

# 5. 查看训练过程曲线（可选）
tensorboard --logdir=runs
```

## 实验结果

| 模型 | 测试集 Top-1 | Macro-F1 |
|---|---|---|
| Baseline（ResNet18） | 89.86% | 0.8985 |
| Label Smoothing（α=0.1） | **90.58%** | **0.9050** |

Label Smoothing 组 Top-1（+0.72pp）与 Macro-F1（+0.0065）均优于 Baseline，验证集最优准确率略低（92.93% vs 93.66%）但测试集指标更高，说明其泛化更稳定、过拟合程度更低。详细分析见《实验报告_宠物分类.docx》。
