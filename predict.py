"""
手势识别实时预测脚本
使用摄像头 + MediaPipe + CNN
"""

import os
import cv2
import mediapipe as mp
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

# ============== CNN 模型 ==============
class GestureCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GestureCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
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


# ============== 预测器 ==============
class GesturePredictor:
    def __init__(self, model_path):
        self.model = GestureCNN(NUM_CLASSES).to(DEVICE)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            self.model.eval()
            print(f"✓ 模型加载成功: {model_path}")
        else:
            print(f"✗ 模型文件不存在: {model_path}")
            print("请先运行 train.py collect 采集数据，再运行 train.py train 训练模型")
            return
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 预测结果平滑
        self.predictions = []
        self.smoothing_window = 5
    
    def predict(self, hand_img):
        """预测单张手势图像"""
        img = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))
        img = self.transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(img)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        return predicted.item(), confidence.item(), probs.cpu().numpy()[0]
    
    def smooth_predict(self, pred, conf):
        """平滑预测结果"""
        self.predictions.append((pred, conf))
        if len(self.predictions) > self.smoothing_window:
            self.predictions.pop(0)
        
        # 取最常见的预测
        from collections import Counter
        preds = [p[0] for p in self.predictions]
        most_common = Counter(preds).most_common(1)[0][0]
        
        # 取平均置信度
        avg_conf = sum([p[1] for p in self.predictions]) / len(self.predictions)
        
        return most_common, avg_conf
    
    def run(self):
        """运行实时预测"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("无法打开摄像头！")
            return
        
        print("\n手势识别实时预测")
        print("按 'q' 退出")
        print("按 'r' 重置预测历史")
        print("-" * 40)
        
        current_gesture = None
        current_confidence = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 镜像
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            gesture_text = "未检测到手"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制关键点
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # 裁剪手部区域
                    h, w, _ = frame.shape
                    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 20
                    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 20
                    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 20
                    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 20
                    
                    x_min, x_max = max(0, x_min), min(w, x_max)
                    y_min, y_max = max(0, y_min), min(h, y_max)
                    
                    if x_max > x_min and y_max > y_min:
                        hand_img = frame[y_min:y_max, x_min:x_max]
                        
                        if hand_img.size > 0:
                            pred, conf = self.predict(hand_img)
                            pred, conf = self.smooth_predict(pred, conf)
                            
                            current_gesture = pred
                            current_confidence = conf
                            
                            gesture_text = f"手势: {pred} ({conf*100:.1f}%)"
            
            # 显示结果
            cv2.putText(frame, gesture_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            # 显示置信度条
            if current_gesture is not None:
                bar_width = int(200 * current_confidence)
                cv2.rectangle(frame, (10, 60), (10 + bar_width, 80), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 60), (210, 80), (255, 255, 255), 2)
            
            # 显示手势 0-9
            cv2.putText(frame, "0  1  2  3  4  5  6  7  8  9",
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Gesture Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.predictions = []
                current_gesture = None
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    predictor = GesturePredictor(MODEL_PATH)
    predictor.run()
