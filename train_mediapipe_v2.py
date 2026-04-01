"""
手势识别训练脚本 - 优化版 v2
- 63维特征 (x, y, z)
- 关键点距离/角度特征
- 增强数据 (缩放、旋转、噪声)
- 更深的 MLP
"""

import os
import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.core.image import Image as MPImage
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

# ============== 配置 ==============
DATA_DIR = "gesture_data_skeleton"
NUM_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 50)
print("手势识别 - 优化版 v2 (63维 + 几何特征)")
print("=" * 50)
print("\n按 Q 退出，按 SPACE 保存当前骨架")


# ============== MediaPipe ==============
def create_hand_landmarker():
    base_options = BaseOptions(model_asset_path='hand_landmarker.task')
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return HandLandmarker.create_from_options(options)


def extract_features(frame, hand_landmarker):
    """
    提取特征:
    - 63维: 21点 x,y,z 坐标
    - 几何特征: 指尖到掌心距离、关键角度、指尖间距离
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
    
    if not result.hand_landmarks:
        return None, None
    
    landmarks = result.hand_landmarks[0]
    
    # === 基础坐标 (63维) ===
    coords = []
    for lm in landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    coords = np.array(coords, dtype=np.float32)  # (63,)
    
    # === 几何特征 ===
    features = []
    
    # 1. 指尖到掌心距离 (8个指尖 - 掌心)
    wrist = landmarks[0]  # 掌心
    fingertips = [4, 8, 12, 16, 20]  # 5个指尖索引
    for idx in fingertips:
        tip = landmarks[idx]
        dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
        features.append(dist)
    
    # 2. 指尖间距离 (两两组合)
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            t1 = landmarks[fingertips[i]]
            t2 = landmarks[fingertips[j]]
            dist = np.sqrt((t1.x - t2.x)**2 + (t1.y - t2.y)**2 + (t1.z - t2.z)**2)
            features.append(dist)
    
    # 3. 指节角度 (用向量叉积近似)
    # 食指: MCP(5), PIP(6), DIP(7), TIP(8)
    for chain in [[5,6,7], [9,10,11], [13,14,15], [17,18,19]]:  # 4个手指的3个关节
        for i in range(len(chain)-2):
            p1 = np.array([landmarks[chain[i]].x, landmarks[chain[i]].y, landmarks[chain[i]].z])
            p2 = np.array([landmarks[chain[i+1]].x, landmarks[chain[i+1]].y, landmarks[chain[i+1]].z])
            p3 = np.array([landmarks[chain[i+2]].x, landmarks[chain[i+2]].y, landmarks[chain[i+2]].z])
            
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            features.append(cos_angle)
    
    # 4. 手掌宽度 (食指MCP到小指MCP距离)
    palm_width = np.sqrt(
        (landmarks[5].x - landmarks[17].x)**2 +
        (landmarks[5].y - landmarks[17].y)**2 +
        (landmarks[5].z - landmarks[17].z)**2
    )
    features.append(palm_width)
    
    # 5. 手掌高度 (掌心到中指MCP)
    palm_height = np.sqrt(
        (landmarks[0].x - landmarks[9].x)**2 +
        (landmarks[0].y - landmarks[9].y)**2 +
        (landmarks[0].z - landmarks[9].z)**2
    )
    features.append(palm_height)
    
    geom_features = np.array(features, dtype=np.float32)  # 约20维
    
    # 归一化坐标 (减去掌心，以掌心为原点)
    norm_coords = coords.copy()
    norm_coords[0::3] -= landmarks[0].x  # x - wrist_x
    norm_coords[1::3] -= landmarks[0].y  # y - wrist_y
    norm_coords[2::3] -= landmarks[0].z  # z - wrist_z
    
    # 合并特征
    combined = np.concatenate([norm_coords, geom_features])
    
    return combined, result.hand_landmarks[0]


def get_hand_info(frame, hand_landmarker):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
    
    if not result.hand_landmarks:
        return None, None
    
    h, w = frame.shape[:2]
    landmarks = result.hand_landmarks[0]
    
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    margin = 0.1
    
    x1 = int(max(0, (min(xs) - margin) * w))
    y1 = int(max(0, (min(ys) - margin) * h))
    x2 = int(min(w, (max(xs) + margin) * w))
    y2 = int(min(h, (max(ys) + margin) * h))
    
    return (x1, y1, x2 - x1, y2 - y1), result.hand_landmarks[0]


def draw_skeleton(frame, hand_landmarks):
    h, w = frame.shape[:2]
    connections = HandLandmarksConnections.HAND_CONNECTIONS
    
    for conn in connections:
        s = hand_landmarks[conn.start]
        e = hand_landmarks[conn.end]
        sx, sy = int(s.x * w), int(s.y * h)
        ex, ey = int(e.x * w), int(e.y * h)
        cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 0), 2)
    
    for lm in hand_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    
    return frame


# ============== 数据采集 ==============
def collect_data(label, num_samples=150):
    label_dir = os.path.join(DATA_DIR, str(label))
    os.makedirs(label_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return
    
    hand_landmarker = create_hand_landmarker()
    collected = 0
    last_save_time = 0
    
    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        features, hand_landmarks = extract_features(frame, hand_landmarker)
        bbox, _ = get_hand_info(frame, hand_landmarker)
        
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            if hand_landmarks:
                draw_skeleton(display, hand_landmarks)
        
        cv2.putText(display, f"Gesture {label}: {collected}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Feature dim: {features.shape[0] if features is not None else 0}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, "SPACE=save  Q=quit", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if features is not None:
            cv2.putText(display, "Hand detected", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No hand", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Camera', display)
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if features is not None:
                now = int(time.time() * 1000)
                if now - last_save_time > 200:
                    np.save(os.path.join(label_dir, f"{now}.npy"), features)
                    collected += 1
                    last_save_time = now
                    print(f"已保存 {collected}/{num_samples}")
            else:
                print("未检测到手，请重试")
    
    hand_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"手势 {label} 采集完成！共 {collected} 张")


def auto_collect_all():
    for i in range(NUM_CLASSES):
        print(f"\n{'='*50}")
        print(f"【手势 {i}】采集")
        print(f"{'='*50}")
        input("→ 将手势摆好，按回车开始采集...")
        collect_data(i, num_samples=150)


# ============== 数据集 ==============
class GestureDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.augment = augment
        self.data = []
        self.labels = []
        
        for label in range(NUM_CLASSES):
            label_path = os.path.join(data_dir, str(label))
            if os.path.exists(label_path):
                for fname in os.listdir(label_path):
                    if fname.endswith('.npy'):
                        self.data.append(os.path.join(label_path, fname))
                        self.labels.append(label)
        
        print(f"加载了 {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = np.load(self.data[idx])
        
        if self.augment:
            features = self.augment_features(features)
        
        features = torch.FloatTensor(features)
        label = torch.LongTensor([self.labels[idx]])[0]
        return features, label
    
    def augment_features(self, features):
        """数据增强 - 84维版本 (63 coords + 21 geom)"""
        # 缩放 (0.9 ~ 1.1)
        scale = np.random.uniform(0.9, 1.1)
        features = features * scale
        
        # 坐标部分加噪声 (63维)
        noise_coords = np.random.randn(63) * 0.01
        # 几何特征部分加噪声 (21维)
        noise_geom = np.random.randn(21) * 0.03
        noise = np.concatenate([noise_coords, noise_geom])
        features = features + noise
        
        return features


# ============== 模型 ==============
class GestureMLP(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


# ============== 训练 ==============
def train_model():
    print(f"使用设备: {DEVICE}")
    
    dataset = GestureDataset(DATA_DIR, augment=True)
    if len(dataset) == 0:
        print("没有找到训练数据！请先采集数据")
        return
    
    # 获取特征维度
    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[0]
    print(f"特征维度: {input_dim}")
    
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=BATCH_SIZE
    )
    
    model = GestureMLP(input_dim=input_dim, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for features, labels in pbar:
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/max(1,total//BATCH_SIZE):.4f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%  LR = {lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_v2.pth')
            print(f"  ✓ 保存最佳模型: {best_acc:.2f}%")
    
    print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python train_mediapipe_v2.py collect      - 采集数据")
        print("  python train_mediapipe_v2.py train        - 训练模型")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "collect":
        if len(sys.argv) >= 3:
            collect_data(int(sys.argv[2]))
        else:
            auto_collect_all()
    elif cmd == "train":
        train_model()
    else:
        print(f"未知命令: {cmd}")
