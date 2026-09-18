# 基于深度学习的牛津宠物细粒度图像分类（Oxford-IIIT Pet，37 类）

科研项目组 72 小时考核的交付仓库。以 ImageNet 预训练 ResNet-18 为基线，完成 Oxford-IIIT Pet 37 类猫、狗品种的细粒度分类，并做了一组单变量消融实验（标准交叉熵 vs. Label Smoothing），包含训练日志、混淆矩阵与 Grad-CAM 可视化。

## 1. 数据集与预处理

| 项目 | 说明 |
|---|---|
| 数据集 | Oxford-IIIT Pet，37 类，官方 trainval 共 3680 张（torchvision 默认划分） |
| 划分方式 | 分层抽样（Stratified）70:15:15，seed=42 → 训练 2576 / 验证 552 / 测试 552 |
| 训练集预处理 | Resize(256) + RandomCrop(224) + RandomHorizontalFlip + Normalize |
| 验证/测试预处理 | Resize(256) + CenterCrop(224) + Normalize（确定性，不含随机增强） |
| 标签 | torchvision 已转成 0~36，可直接配合 nn.CrossEntropyLoss() |

数据集体积约 800 MB，不随仓库提交。脚本内 `download=True` 会自动下载到 `data/`；若官方源太慢，可手动下载 `images.tar.gz` / `annotations.tar.gz` 解压成 `data/oxford-iiit-pet/{images,annotations}/` 后再运行。

## 2. 环境要求

- Python 3.8+，PyTorch 2.x，torchvision
- scikit-learn、matplotlib、tensorboard、Pillow
- 实测硬件：RTX 4060 Laptop 8 GB / Windows 11（CPU 也可运行，仅速度较慢）

## 3. 目录结构

```
pet-classifier/
├── pet_data.py          # 数据加载、分层划分与 Transform
├── day2_ls.py           # Label Smoothing 训练（日志→runs/pet_ls，模型→ls_best.pth）
├── day2_evaluate.py     # 测试集评估：Top-1 / Macro-F1 / 混淆矩阵
├── day2_curves.py       # 由 TensorBoard 日志绘制训练曲线
├── day2_gradcam.py      # Grad-CAM 可视化
├── requirements.txt     # 依赖清单
├── README.md            # 项目说明与一键复现命令
├── runs/                # TensorBoard 日志（Baseline 与 Label Smoothing 两组）
├── data/                # 数据集（自动下载，不随仓库提交）
└── 实验报告_宠物分类.docx # 考核技术报告
```

## 4. 一键复现

以下命令在项目根目录执行，脚本自动选择 GPU/CPU：

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 训练（Label Smoothing，10 轮，最优模型保存为 ls_best.pth）
python day2_ls.py

# 3) 测试集评估（输出 Top-1 / Macro-F1，生成混淆矩阵 confusion_ls.png）
python day2_evaluate.py

# 4) 绘制 Baseline vs Label Smoothing 训练曲线（生成 training_curves.png）
python day2_curves.py

# 5) Grad-CAM 可视化（生成 gradcam_ls.png）
python day2_gradcam.py

# 6) 查看训练曲线（TensorBoard，浏览器打开 http://localhost:6006）
tensorboard --logdir=runs
```

## 5. 实验结果

| 模型 | 测试集 Top-1 | Macro-F1 |
|---|---|---|
| Baseline（ResNet18） | 89.86% | 0.8985 |
| Label Smoothing（α=0.1） | **90.58%** | **0.9050** |

Label Smoothing 组 Top-1（+0.72pp）与 Macro-F1（+0.0065）均优于 Baseline：软标签抑制模型过度自信，在细粒度分类（类间外观高度相似、易过拟合）场景下带来更稳定的泛化收益。详细分析见《实验报告_宠物分类.docx》。
