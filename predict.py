"""
手势识别实时预测脚本
支持手部区域自动检测
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# ============== 配置 ==============
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 10
IMG_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== 手部检测 ==============
def detect_hand(frame):
    """检测手部区域，返回 (x, y, w, h) 或 None"""
    
    # 方法1: 肤色检测 + 轮廓分析
    def skin_detect(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 肤色范围
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # 去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 高斯模糊
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        return mask
    
    mask = skin_detect(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # 找最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        
        # 过滤小区域
        if area > 5000:
            x, y, w, h = cv2.boundingRect(max_contour)
            # 确保区域合理
            if w > 30 and h > 30 and w < frame.shape[1] * 0.9 and h < frame.shape[0] * 0.9:
                return (x, y, w, h)
    
    return None


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


# ============== 预测器 ==============
class GesturePredictor:
    def __init__(self, model_path):
        self.model = GestureCNN(NUM_CLASSES).to(DEVICE)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.eval()
            print(f"✓ 模型加载成功")
        else:
            print(f"✗ 模型不存在: {model_path}")
            return
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.predictions = []
        self.window = 5
    
    def predict(self, hand_img):
        img = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))
        img = self.transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            out = self.model(img)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
        
        return pred.item(), conf.item()
    
    def smooth(self, pred, conf):
        self.predictions.append((pred, conf))
        if len(self.predictions) > self.window:
            self.predictions.pop(0)
        
        from collections import Counter
        most_common = Counter([p[0] for p in self.predictions]).most_common(1)[0][0]
        avg_conf = sum([p[1] for p in self.predictions]) / len(self.predictions)
        return most_common, avg_conf
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头！")
            return
        
        print("\n手势识别 - 手放到画面任意位置 - 按 Q 退出\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            # 检测手部
            roi = detect_hand(frame)
            
            gesture_text = "将手放入画面"
            conf = 0
            
            if roi:
                x, y, w, h = roi
                
                # 扩大一点区域
                margin = 10
                x, y = max(0, x - margin), max(0, y - margin)
                w, h = min(frame.shape[1] - x, w + 2*margin), min(frame.shape[0] - y, h + 2*margin)
                
                # 画框
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 裁剪手部区域
                hand = frame[y:y+h, x:x+w]
                if hand.size > 0:
                    pred, conf = self.predict(hand)
                    pred, conf = self.smooth(pred, conf)
                    gesture_text = f"手势: {pred} ({conf*100:.1f}%)"
            else:
                # 没检测到时显示提示
                cv2.putText(display, "将手放入画面中...", (display.shape[1]//2 - 100, display.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 显示结果
            cv2.putText(display, gesture_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            if conf > 0:
                bar_w = int(200 * conf)
                cv2.rectangle(display, (10, 60), (10+bar_w, 80), (0, 255, 0), -1)
                cv2.rectangle(display, (10, 60), (210, 80), (255, 255, 255), 2)
            
            cv2.putText(display, "0  1  2  3  4  5  6  7  8  9",
                       (10, display.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Gesture Recognition", display)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GesturePredictor(MODEL_PATH).run()
