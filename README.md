# Gesture Recognition - MediaPipe v2

基于 MediaPipe Hands 的手势数字识别（0-9），使用关键点几何特征 + MLP 神经网络。

## 项目结构

```
GR/
├── requirements.txt              # 依赖
├── train_mediapipe_v2.py        # 训练脚本 (数据采集 + 训练)
├── predict_mediapipe_v2.py      # 实时预测脚本
├── predict_mediapipe.py         # 中间版本
├── train_mediapipe.py           # 中间版本
├── predict.py                   # 旧版 (CNN 图像方案)
├── train.py                      # 旧版 (CNN 图像方案)
├── best_model_v2.pth            # 训练好的模型 (v2, 不上传)
├── best_model.pth               # 旧版模型 (不上传)
├── gesture_data/                # 旧版图像数据 (不上传)
├── gesture_data_skeleton/       # 骨架关键点数据 (不上传)
├── hand_landmarker.task         # MediaPipe 模型 (不上传)
└── README.md
```

## 环境配置

```bash
pip install opencv-python numpy torch torchvision mediapipe scikit-learn tqdm pillow
```

## 使用方法

### 1. 数据采集

```bash
# 采集所有手势 0-9
python train_mediapipe_v2.py collect

# 采集单个手势
python train_mediapipe_v2.py collect 5
```

- 按空格键保存样本
- 按 `q` 退出
- 建议采集 150 张/手势

### 2. 训练模型

```bash
python train_mediapipe_v2.py train
```

### 3. 实时预测

```bash
python predict_mediapipe_v2.py [模型路径] [特征维度]
# 例如
python predict_mediapipe_v2.py best_model_v2.pth 84
```

## 特征工程 (v2)

输入特征 **84 维**：

| 特征类型 | 维度 | 说明 |
|---------|------|------|
| 归一化坐标 | 63 | 21 关键点 × 3 (x, y, z)，以掌心为原点 |
| 指尖-掌心距离 | 5 | 5 个指尖到掌心的欧氏距离 |
| 指尖间距离 | 10 | C(5,2) 两两指尖距离 |
| 指节角度 | 8 | 4 个手指链 × 2 个角度 (cos 值) |
| 手掌宽/高 | 2 | 手掌宽度和高度 |

## 模型架构 (v2)

```
GestureMLP (84 → 10)
├── Linear(84, 256) + BN + ReLU + Dropout(0.3)
├── Linear(256, 512) + BN + ReLU + Dropout(0.3)
├── Linear(512, 256) + BN + ReLU + Dropout(0.2)
├── Linear(256, 128) + ReLU + Dropout(0.1)
└── Linear(128, 10)
```

## 数据增强

- 随机缩放 (0.9 ~ 1.1)
- 坐标噪声 (σ=0.01)
- 几何特征噪声 (σ=0.03)

## 版本历史

- **v2**: MediaPipe 关键点 + 几何特征 + MLP (当前版本)
- **v1**: MediaPipe 关键点 (过渡版本)
- **原始版**: CNN + 肤色检测 + 人脸排除

## 参考

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [PyTorch](https://pytorch.org/)
