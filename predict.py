"""
手势识别实时预测脚本
肤色检测 + 人脸排除
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

# ============== 加载人脸检测器 ==============
face_cascade = None
try:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    print("✓ 人脸检测器加载成功")
except:
    print("✗ 人脸检测器加载失败")

# ============== 手部检测 ==============
def detect_hand(frame):
    """检测手部区域，排除人脸"""
    
    def skin_mask(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 肤色范围
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
    
    mask = skin_mask(frame)
    
    # 人脸区域 - 需要排除
    face_rects = []
    if face_cascade:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (fx, fy, fw, fh) in faces:
            # 扩展人脸区域（脖子、肩膀也要排除）
            margin = 30
            fx2 = max(0, fx - margin)
            fy2 = max(0, fy - margin)
            fx2_end = min(frame.shape[1], fx + fw + margin)
            fy2_end = min(frame.shape[0], fy + fh + margin + 50)  # 往下扩展
            face_rects.append((fx2, fy2, fx2_end - fx2, fy2_end - fy2))
    
    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 过滤掉人脸区域
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
            # 计算重叠
            overlap_x = max(x, fx)
            overlap_y = max(y, fy)
            overlap_x_end = min(x + w, fx + fw)
            overlap_y_end = min(y + h, fy + fh)
            
            if overlap_x < overlap_x_end and overlap_y < overlap_y_end:
                overlap_area = (overlap_x_end - overlap_x) * (overlap_y_end - overlap_y)
                overlap_ratio = overlap_area / (w * h)
                if overlap_ratio > 0.3:  # 30%以上重叠认为是人脸
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
    
    # 扩展边界
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(frame.shape[1] - x, w + 2 * margin)
    h = min(frame.shape[0] - y, h + 2 * margin)
    
    return (x, y, w, h)


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
        self.last_roi = None
    
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
        
        print("\n手势识别 - 按 Q 退出\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            roi = detect_hand(frame)
            
            if roi is None and self.last_roi is not None:
                roi = self.last_roi
            
            gesture_text = "将手放入画面"
            conf = 0
            
            if roi:
                x, y, w, h = roi
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                hand = frame[y:y+h, x:x+w]
                if hand.size > 0 and w > 20 and h > 20:
                    pred, conf = self.predict(hand)
                    pred, conf = self.smooth(pred, conf)
                    gesture_text = f"手势: {pred} ({conf*100:.1f}%)"
                    self.last_roi = roi
            else:
                self.last_roi = None
            
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
