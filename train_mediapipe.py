"""
手势识别训练脚本 - MediaPipe 骨架版本 (v0.10+)
泛化能力强，不受光照/背景影响
"""

import os
import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision.core.image import Image as MPImage
from mediapipe.tasks.python.vision.core.image import ImageFormat
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
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 静默 MediaPipe 日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 50)
print("手势识别 - MediaPipe 骨架版本 (v0.10+)")
print("=" * 50)
print("\n按 Q 退出，按 SPACE 保存当前骨架")


# ============== MediaPipe 处理函数 ==============
def create_hand_landmarker():
    """创建 HandLandmarker 实例"""
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


def extract_skeleton(frame, hand_landmarker):
    """
    从画面中提取21个手部关节点坐标
    返回 42 维特征 (21点 x, y) 或 None
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
    result = hand_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
    
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        coords = []
        for lm in landmarks:
            coords.append(lm.x)
            coords.append(lm.y)
        return np.array(coords, dtype=np.float32)
    return None


def get_hand_info(frame, hand_landmarker):
    """获取手的边界框和骨架数据"""
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
    """绘制骨架"""
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
    """采集骨架数据"""
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
        
        skeleton = extract_skeleton(frame, hand_landmarker)
        bbox, hand_landmarks = get_hand_info(frame, hand_landmarker)
        
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            if hand_landmarks:
                draw_skeleton(display, hand_landmarks)
        
        cv2.putText(display, f"Gesture {label}: {collected}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=save  Q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if skeleton is not None:
            cv2.putText(display, "Hand detected", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No hand", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Camera', display)
        
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if skeleton is not None:
                now = int(time.time() * 1000)
                if now - last_save_time > 200:
                    np.save(os.path.join(label_dir, f"{now}.npy"), skeleton)
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
    """引导式采集所有手势"""
    for i in range(NUM_CLASSES):
        print(f"\n{'='*50}")
        print(f"【手势 {i}】采集")
        print(f"{'='*50}")
        input("→ 将手势摆好，按回车开始采集...")
        collect_data(i, num_samples=150)


# ============== 数据集 ==============
class SkeletonDataset(Dataset):
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
        
        print(f"加载了 {len(self.data)} 个骨架样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        skeleton = np.load(self.data[idx])  # (42,)
        
        if self.augment:
            noise = np.random.randn(42) * 0.01
            skeleton = skeleton + noise
        
        skeleton = torch.FloatTensor(skeleton)
        label = torch.LongTensor([self.labels[idx]])[0]
        return skeleton, label


# ============== 模型 ==============
class SkeletonMLP(nn.Module):
    def __init__(self, input_dim=42, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


# ============== 训练 ==============
def train_model():
    print(f"使用设备: {DEVICE}")
    
    dataset = SkeletonDataset(DATA_DIR, augment=True)
    if len(dataset) == 0:
        print("没有找到训练数据！请先采集骨架数据")
        return
    
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=BATCH_SIZE
    )
    
    sample_input = dataset[0][0]
    input_dim = sample_input.shape[0]
    print(f"输入维度: {input_dim}")
    
    model = SkeletonMLP(input_dim=input_dim, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for skeletons, labels in pbar:
            skeletons, labels = skeletons.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(skeletons)
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
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for skeletons, labels in val_loader:
                skeletons, labels = skeletons.to(DEVICE), labels.to(DEVICE)
                outputs = model(skeletons)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model_skeleton.pth')
            print(f"  ✓ 保存最佳模型: {best_acc:.2f}%")
    
    print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python train_mediapipe.py collect      - 采集所有手势")
        print("  python train_mediapipe.py collect 5   - 采集手势 5")
        print("  python train_mediapipe.py train        - 训练模型")
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
