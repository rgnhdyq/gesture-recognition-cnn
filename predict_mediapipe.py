"""
手势识别实时预测脚本 - MediaPipe 骨架版本 (v0.10+)
泛化能力强，不受光照/背景影响
"""

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision.core.image import Image as MPImage
from mediapipe.tasks.python.vision.core.image import ImageFormat
import torch
import torch.nn as nn
from collections import Counter
import time
import os

# ============== 配置 ==============
MODEL_PATH = "best_model_skeleton.pth"
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


# ============== 预测器 ==============
class SkeletonPredictor:
    def __init__(self, model_path):
        # 加载 PyTorch 模型
        self.model = SkeletonMLP(input_dim=42, num_classes=NUM_CLASSES).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print(f"✓ PyTorch 模型加载成功 ({MODEL_PATH})")
        
        # 创建 MediaPipe HandLandmarker
        base_options = BaseOptions(model_asset_path='hand_landmarker.task')
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_landmarker = HandLandmarker.create_from_options(options)
        print("✓ MediaPipe HandLandmarker 初始化成功")
        
        self.predictions = []
        self.window = 8
    
    def extract_skeleton(self, frame):
        """从画面提取21个关节点坐标"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        result = self.hand_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
        
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            coords = []
            for lm in landmarks:
                coords.extend([lm.x, lm.y])
            return np.array(coords, dtype=np.float32), result.hand_landmarks[0]
        return None, None
    
    def predict(self, skeleton):
        """单帧预测"""
        with torch.no_grad():
            x = torch.FloatTensor(skeleton).unsqueeze(0).to(DEVICE)
            out = self.model(x)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
        return pred.item(), conf.item()
    
    def smooth(self, pred, conf):
        """平滑预测"""
        self.predictions.append((pred, conf))
        if len(self.predictions) > self.window:
            self.predictions.pop(0)
        
        most_common = Counter([p[0] for p in self.predictions]).most_common(1)[0][0]
        avg_conf = sum([p[1] for p in self.predictions]) / len(self.predictions)
        return most_common, avg_conf
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头！")
            return
        
        print("\n手势识别 (MediaPipe 骨架版 v0.10+)")
        print("按 Q 退出\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            skeleton, hand_landmarks = self.extract_skeleton(display)
            
            gesture_text = "请将手放入画面"
            conf = 0
            
            if skeleton is not None and hand_landmarks is not None:
                # 绘制骨架
                h, w = display.shape[:2]
                connections = HandLandmarksConnections.HAND_CONNECTIONS
                for conn in connections:
                    s = hand_landmarks[conn.start]
                    e = hand_landmarks[conn.end]
                    sx, sy = int(s.x * w), int(s.y * h)
                    ex, ey = int(e.x * w), int(e.y * h)
                    cv2.line(display, (sx, sy), (ex, ey), (0, 255, 0), 2)
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(display, (x, y), 4, (0, 0, 255), -1)
                
                pred, conf = self.predict(skeleton)
                pred, conf = self.smooth(pred, conf)
                gesture_text = f"手势: {pred}  ({conf*100:.1f}%)"
            
            cv2.putText(display, gesture_text, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            if conf > 0:
                bar_w = int(200 * conf)
                cv2.rectangle(display, (10, 60), (10+bar_w, 80), (0, 255, 0), -1)
                cv2.rectangle(display, (10, 60), (210, 80), (255, 255, 255), 2)
            
            cv2.putText(display, "0  1  2  3  4  5  6  7  8  9",
                       (10, display.shape[0]-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("MediaPipe Gesture Recognition", display)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cap.release()
        self.hand_landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    SkeletonPredictor(MODEL_PATH).run()
