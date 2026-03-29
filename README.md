# 手势识别 - 深度学习项目

基于 CNN 的手势数字识别（0-9），使用 MediaPipe 进行手部检测。

## 项目结构

```
GR/
├── requirements.txt    # 依赖包
├── train.py          # 训练脚本（数据采集 + 模型训练）
├── predict.py        # 实时预测脚本
├── best_model.pth    # 训练好的模型（训练后生成）
├── gesture_data/     # 手势数据目录（采集后生成）
│   ├── 0/            # 数字0的手势
│   ├── 1/
│   ├── ...
│   └── 9/
└── README.md
```

## 环境配置

```bash
# 1. 创建 conda 环境（推荐）
conda create -n gesture python=3.9
conda activate gesture

# 2. 安装 PyTorch（根据你的 CUDA 版本）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision

# 3. 安装其他依赖
pip install -r requirements.txt
```

## 使用步骤

### 1. 数据采集

```bash
# 采集所有手势 0-9
python train.py collect

# 或者采集单个手势（测试用）
python train.py collect 5  # 只采集数字5
```

**采集说明：**
- 按空格键保存图片
- 按 `q` 跳过当前手势
- 建议采集 100-200 张/手势
- 确保光线充足，手部清晰

### 2. 训练模型

```bash
python train.py train
```

训练完成后：
- 最佳模型保存为 `best_model.pth`
- 验证准确率会在终端显示

### 3. 实时预测

```bash
python predict.py
```

**操作说明：**
- 按 `q` 退出
- 按 `r` 重置预测历史
- 确保摄像头正常工作

## 模型架构

```
GestureCNN
├── Conv2d(3, 32) + BN + ReLU + MaxPool
├── Conv2d(32, 64) + BN + ReLU + MaxPool
├── Conv2d(64, 128) + BN + ReLU + MaxPool
├── Conv2d(128, 256) + BN + ReLU + MaxPool
├── Flatten
├── Dropout(0.5)
├── Linear(4096, 512) + ReLU + Dropout(0.3)
└── Linear(512, 10)
```

## 数据增强

训练时使用以下增强：
- 随机旋转 (±15°)
- 随机水平翻转
- 颜色抖动 (亮度±20%, 对比度±20%)

## 常见问题

### Q: 摄像头无法打开？
```python
# 修改 predict.py 中的摄像头索引
cap = cv2.VideoCapture(1)  # 尝试索引 1, 2...
```

### Q: 训练时显存不足？
```python
# 减小 batch size（train.py 中修改）
BATCH_SIZE = 16  # 原来 32
```

### Q: 模型预测不准？
- 增加训练数据量
- 确保采集时手势多样性（不同角度、距离）
- 调整 MediaPipe 的 `min_detection_confidence`

## 参考资料

- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [LeNet-5 Architecture](http://yann.lecun.com/exdb/lenet/)
