"""
手势识别训练脚本
手势数字 0-9
"""

import os
import cv2
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
NUM_CLASSES = 10
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OpenCV 设置
cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)

print("按 Q 退出，按 SPACE 保存当前画面")


def collect_data(label, num_samples=100):
    """采集手势数据"""
    label_dir = os.path.join(DATA_DIR, str(label))
    os.makedirs(label_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return
    
    collected = 0
    
    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # 中心区域作为默认手部区域
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = min(w, h) // 3
        x, y = cx - size//2, cy - size//2
        
        # 画框
        display = frame.copy()
        cv2.rectangle(display, (x, y), (x+size, y+size), (0, 255, 0), 2)
        cv2.putText(display, f"Gesture {label}: {collected}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=save", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Camera', display)
        cv2.imshow('Preview', cv2.resize(frame[y:y+size, x:x+size], (300, 300)))
        
        # 处理按键事件
        key = cv2.waitKey(100) & 0xFF  # 100ms 超时，避免完全卡死
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            hand_img = frame[y:y+size, x:x+size]
            hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(label_dir, f"{int(time.time()*1000)}.jpg"), hand_img)
            collected += 1
            print(f"已保存 {collected}/{num_samples}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"手势 {label} 采集完成！共 {collected} 张")


def auto_collect_all():
    """采集所有手势 0-9"""
    for i in range(NUM_CLASSES):
        print(f"\n{'='*50}")
        print(f"手势 {i} - 按 SPACE 保存，按 Q 退出")
        print(f"{'='*50}")
        collect_data(i, num_samples=150)


# ============== 数据集 ==============
class GestureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
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
        
        print(f"加载了 {len(self.images)} 张图片")
    
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
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))


# ============== 训练 ==============
def train_model():
    print(f"使用设备: {DEVICE}")
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = GestureDataset(DATA_DIR, train_transform)
    
    if len(dataset) == 0:
        print("没有找到训练数据！请先采集数据")
        return
    
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), 
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), 
                            batch_size=BATCH_SIZE)
    
    model = GestureCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{total_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        scheduler.step()
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}: Val Acc={val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ 保存最佳模型: {best_acc:.2f}%")
    
    print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python train.py collect     - 采集所有手势")
        print("  python train.py collect 5   - 采集手势 5")
        print("  python train.py train       - 训练模型")
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
