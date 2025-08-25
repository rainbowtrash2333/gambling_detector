# gambling_detector

这段代码实现了一个 基于 PyTorch 的 CNN 赌博网站检测器，核心功能是通过卷积神经网络（CNN）模型识别网站截图是否属于赌博网站，并能自动分类保存。整体逻辑可以分为以下几个部分：

---

### 1. 数据集管理 `GamblingDataset`

* **训练模式 (train)**：

  * 从 `scan_results/train/0` 加载 **非赌博网站截图**（标签=0）。
  * 从 `scan_results/train/1` 加载 **赌博网站截图**（标签=1）。
* **预测模式 (predict)**：

  * 从 `scan_results/screenshots/` 加载待检测图片（不含标签）。
* 支持图像预处理（缩放、增强、归一化），并对异常图片返回占位图。

---

### 2. 卷积神经网络 `GamblingCNN`

* 网络结构类似 **AlexNet**，包含 5 个卷积块和 3 个全连接层：

  * 特征提取层：多层卷积 + ReLU 激活 + 最大池化。
  * 自适应平均池化 (6×6)。
  * 分类器：两层全连接层（含 Dropout 防止过拟合）+ 最终输出层（2 类：赌博 / 非赌博）。

---

### 3. 检测器核心类 `GamblingDetector`

功能模块：

* **训练 (train)**

  * 使用 `CrossEntropyLoss` 作为损失函数，`Adam` 优化器，带 `StepLR` 学习率衰减。
  * 每轮训练统计 loss 和准确率，保存最佳模型到 `gambling_detector_model.pth`。

* **预测 (predict)**

  * 加载最佳模型，对 `scan_results/screenshots/` 里的截图进行推理。
  * 使用 softmax 概率判断赌博类别（阈值默认 0.7）。
  * 检测出的赌博网站截图复制到 `scan_results/test/`，并在文件名里附加置信度。
  * 输出统计报告（总数、检测数、检测率）。

---


## ✅ 基本运行命令

### 1. 训练模型

```bash
python3 gambling_detector.py --mode train --epochs 50 --batch_size 32 --learning_rate 0.001 --gpu
```

说明：

* `--mode train` → 仅训练模型。
* `--epochs 50` → 训练 50 个 epoch（可改）。
* `--gpu` → 如果有 CUDA，会用 GPU，否则自动退回 CPU。

训练后的模型保存在：

```
scan_results/models/gambling_detector_model.pth
```

---

### 2. 预测赌博网站

```bash
python3 gambling_detector.py --mode predict --confidence 0.7 --gpu
```

说明：

* `--mode predict` → 仅做预测，不训练。
* `--confidence 0.7` → 置信度阈值，默认 0.7。超过这个值的截图会被判定为赌博网站。
* 预测出的赌博网站图片会被复制到：

```
scan_results/test/
```

文件名里会带上置信度，例如：

```
example_conf0.845.png
```

---

### 3. 先训练再预测（推荐）

```bash
python3 gambling_detector.py --mode both --epochs 30 --batch_size 16 --gpu
```

说明：

* `--mode both` → 先训练模型，然后用训练好的模型跑预测。

---

### 4. 查看已保存的模型

```bash
python3 gambling_detector.py --mode list
```

会输出：

* 模型文件列表（`.pth` 权重文件）。
* 模型大小、保存时间。
* 训练报告（损失、准确率、训练样本数等）。

---

## ✅ 可选参数一览

| 参数                | 作用                                    | 默认值    |
| ----------------- | ------------------------------------- | ------ |
| `--mode`          | 运行模式（`train`/`predict`/`both`/`list`） | `both` |
| `--gpu`           | 启用 GPU                                | 否      |
| `--epochs`        | 训练轮数                                  | 50     |
| `--batch_size`    | 批次大小                                  | 32     |
| `--learning_rate` | 学习率                                   | 0.001  |
| `--confidence`    | 预测置信度阈值                               | 0.7    |
| `--model_path`    | 指定预训练模型路径                             | 无      |

