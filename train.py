"""
手势识别训练脚本
支持肤色检测 + 人脸排除
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

# ============== 人脸检测器 ==============
face_cascade = None
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    print("✓ 人脸检测器加载成功")
except:
    print("✗ 人脸检测器加载失败")

print("按 Q 退出，按 SPACE 保存当前画面")


# ============== 手部检测 ==============
def detect_hand(frame):
    """检测手部区域，排除人脸"""
    
    def skin_mask(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower1 = np.array([0, 20, 70], dtype=np.uint8)
        upper1 = np.array([20, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        lower2 = np.array([165, 20, 70], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        return mask
    
    # 人脸区域
    face_rects = []
    if face_cascade:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (fx, fy, fw, fh) in faces:
            margin = 30
            fx2 = max(0, fx - margin)
            fy2 = max(0, fy - margin)
            fx2_end = min(frame.shape[1], fx + fw + margin)
            fy2_end = min(frame.shape[0], fy + fh + margin + 50)
            face_rects.append((fx2, fy2, fx2_end - fx2, fy2_end - fy2))
    
    mask = skin_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    best_contour = None
    best_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 3000 or area > frame.shape[0] * frame.shape[1] * 0.7:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # 检查是否与人脸重叠
        is_face = False
        for (fx, fy, fw, fh) in face_rects:
            overlap_x = max(x, fx)
            overlap_y = max(y, fy)
            overlap_x_end = min(x + w, fx + fw)
            overlap_y_end = min(y + h, fy + fh)
            
            if overlap_x < overlap_x_end and overlap_y < overlap_y_end:
                overlap_area = (overlap_x_end - overlap_x) * (overlap_y_end - overlap_y)
                overlap_ratio = overlap_area / (w * h)
                if overlap_ratio > 0.3:
                    is_face = True
                    break
        
        if is_face:
            continue
        
        if area > best_area:
            best_area = area
            best_contour = contour
    
    if best_contour is None:
        return None
    
    x, y, w, h = cv2.boundingRect(best_contour)
    
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(frame.shape[1] - x, w + 2 * margin)
    h = min(frame.shape[0] - y, h + 2 * margin)
    
    return (x, y, w, h)


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
        display = frame.copy()
        
        # 检测手部
        roi = detect_hand(frame)
        
        if roi:
            x, y, w, h = roi
            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            hand_img = frame[y:y+h, x:x+w]
            cv2.imshow('Preview', cv2.resize(hand_img, (300, 300)))
        else:
            # 默认中心区域
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            size = min(w, h) // 3
            x, y = cx - size//2, cy - size//2
        
        cv2.putText(display, f"Gesture {label}: {collected}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=save", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow('Camera', display)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(100) & 0xFF == ord(' '):
            if roi:
                x, y, w, h = roi
            else:
                h, w = frame.shape[:2]
                cx, cy = w // 2, h // 2
                size = min(w, h) // 3
                x, y = cx - size//2, cy - size//2
                print("使用中心区域（未检测到手）")
            
            hand_img = frame[y:y+h, x:x+w]
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
        input("按回车开始...")
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
        print("  python train.py train      - 训练模型")
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
