"""
手势识别训练脚本
使用 MediaPipe 采集数据 + CNN 训练
手势数字 0-9
"""

import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

# ============== 配置 ==============
DATA_DIR = "gesture_data"
NUM_CLASSES = 10  # 0-9 数字手势
IMG_SIZE = 64     # 图像大小
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== MediaPipe 配置 ==============
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ============== 数据采集 ==============
def collect_data(label, num_samples=100):
    """采集手势数据"""
    label_dir = os.path.join(DATA_DIR, str(label))
    os.makedirs(label_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    print(f"开始采集手势 {label}，按 'q' 退出...")
    
    collected = 0
    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 镜像显示
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制关键点
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 裁剪手部区域
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 20
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 20
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 20
                
                # 确保在图像范围内
                x_min, x_max = max(0, x_min), min(w, x_max)
                y_min, y_max = max(0, y_min), min(h, y_max)
                
                if x_max > x_min and y_max > y_min:
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    
                    # 每10帧采集一次
                    if collected % 10 == 0 and collected > 0:
                        pass
                    cv2.imshow('Collecting', hand_img)
        
        cv2.putText(frame, f"Gesture {label}: {collected}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Collecting', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格键保存
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                h, w, _ = frame.shape
                x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 20
                x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 20
                y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
                y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 20
                
                x_min, x_max = max(0, x_min), min(w, x_max)
                y_min, y_max = max(0, y_min), min(h, y_max)
                
                if x_max > x_min and y_max > y_min:
                    hand_img = frame[y_min:y_max, x_min:x_max]
                    hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                    cv2.imwrite(os.path.join(label_dir, f"{int(time.time())}.jpg"), hand_img)
                    collected += 1
                    print(f"已保存 {collected}/{num_samples}")
        
        time.sleep(0.05)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"手势 {label} 采集完成！")


def auto_collect_all():
    """自动采集所有手势 0-9"""
    for i in range(NUM_CLASSES):
        print(f"\n{'='*50}")
        print(f"准备采集手势 {i}")
        print("按空格键保存图片，按 'q' 跳过到下一个手势")
        print(f"{'='*50}")
        input("按回车开始...")
        collect_data(i, num_samples=150)


# ============== 数据集 ==============
class GestureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label in range(NUM_CLASSES):
            label_path = os.path.join(data_dir, str(label))
            if os.path.exists(label_path):
                for img_name in os.listdir(label_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.images.append(os.path.join(label_path, img_name))
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, label


# ============== CNN 模型 ==============
class GestureCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GestureCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16 -> 8
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8 -> 4
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============== 训练 ==============
def train_model():
    """训练模型"""
    print(f"使用设备: {DEVICE}")
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    dataset = GestureDataset(DATA_DIR, transform=train_transform)
    
    if len(dataset) == 0:
        print("没有找到训练数据！请先运行 data_collection() 采集数据")
        return
    
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 模型
    model = GestureCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{train_loss/train_total:.4f}', 
                            'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss={train_loss/train_total:.4f}, "
              f"Train Acc={100.*train_correct/train_total:.2f}%, "
              f"Val Acc={val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ 保存最佳模型，准确率: {best_acc:.2f}%")
    
    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")
    return model


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  python train.py collect      - 采集所有手势数据")
        print("  python train.py collect 5    - 采集手势数字 5")
        print("  python train.py train        - 训练模型")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "collect":
        if len(sys.argv) >= 3:
            label = int(sys.argv[2])
            collect_data(label)
        else:
            auto_collect_all()
    elif command == "train":
        train_model()
    else:
        print(f"未知命令: {command}")
