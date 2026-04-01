"""
手势识别实时预测脚本 - 优化版 v2
"""

import cv2
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.vision.core.image import Image as MPImage
from mediapipe.tasks.python.vision.core.image import ImageFormat
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
import torch
import torch.nn as nn
from collections import Counter
import time
import os

# ============== 配置 ==============
MODEL_PATH = "best_model_v2.pth"
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


# ============== 预测器 ==============
class GesturePredictor:
    def __init__(self, model_path, feature_dim):
        self.feature_dim = feature_dim
        self.model = GestureMLP(input_dim=feature_dim, num_classes=NUM_CLASSES).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print(f"✓ 模型加载成功: {MODEL_PATH} (特征维度={feature_dim})")
        
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
        print("✓ MediaPipe 初始化成功")
        
        self.predictions = []
        self.window = 8
    
    def extract_features(self, frame):
        """提取与训练时完全相同的特征"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb)
        result = self.hand_landmarker.detect_for_video(mp_image, int(time.time() * 1000))
        
        if not result.hand_landmarks:
            return None, None
        
        landmarks = result.hand_landmarks[0]
        
        # === 基础坐标 (63维) ===
        coords = []
        for lm in landmarks:
            coords.extend([lm.x, lm.y, lm.z])
        coords = np.array(coords, dtype=np.float32)
        
        # === 几何特征 ===
        features = []
        
        # 1. 指尖到掌心距离 (5个指尖)
        wrist = landmarks[0]
        fingertips = [4, 8, 12, 16, 20]
        for idx in fingertips:
            tip = landmarks[idx]
            dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
            features.append(dist)
        
        # 2. 指尖间距离 (两两组合, C(5,2)=10)
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                t1 = landmarks[fingertips[i]]
                t2 = landmarks[fingertips[j]]
                dist = np.sqrt((t1.x - t2.x)**2 + (t1.y - t2.y)**2 + (t1.z - t2.z)**2)
                features.append(dist)
        
        # 3. 指节角度 (4个手指链, 每个2个角度)
        for chain in [[5,6,7], [9,10,11], [13,14,15], [17,18,19]]:
            for i in range(len(chain)-2):
                p1 = np.array([landmarks[chain[i]].x, landmarks[chain[i]].y, landmarks[chain[i]].z])
                p2 = np.array([landmarks[chain[i+1]].x, landmarks[chain[i+1]].y, landmarks[chain[i+1]].z])
                p3 = np.array([landmarks[chain[i+2]].x, landmarks[chain[i+2]].y, landmarks[chain[i+2]].z])
                v1 = p1 - p2
                v2 = p3 - p2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                features.append(cos_angle)
        
        # 4. 手掌宽度
        palm_width = np.sqrt(
            (landmarks[5].x - landmarks[17].x)**2 +
            (landmarks[5].y - landmarks[17].y)**2 +
            (landmarks[5].z - landmarks[17].z)**2
        )
        features.append(palm_width)
        
        # 5. 手掌高度
        palm_height = np.sqrt(
            (landmarks[0].x - landmarks[9].x)**2 +
            (landmarks[0].y - landmarks[9].y)**2 +
            (landmarks[0].z - landmarks[9].z)**2
        )
        features.append(palm_height)
        
        geom_features = np.array(features, dtype=np.float32)
        
        # 归一化坐标 (以掌心为原点)
        norm_coords = coords.copy()
        norm_coords[0::3] -= landmarks[0].x
        norm_coords[1::3] -= landmarks[0].y
        norm_coords[2::3] -= landmarks[0].z
        
        # 合并
        combined = np.concatenate([norm_coords, geom_features])
        
        return combined, result.hand_landmarks[0]
    
    def predict(self, features):
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
            out = self.model(x)
            prob = torch.softmax(out, dim=1)
            conf, pred = torch.max(prob, 1)
        return pred.item(), conf.item()
    
    def smooth(self, pred, conf):
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
        
        print("\n手势识别 v2 (63维+几何特征)")
        print("按 Q 退出\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            features, hand_landmarks = self.extract_features(display)
            
            gesture_text = "请将手放入画面"
            conf = 0
            
            if features is not None and hand_landmarks is not None:
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
                
                pred, conf = self.predict(features)
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
            
            cv2.imshow("Gesture Recognition v2", display)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        
        cap.release()
        self.hand_landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 2:
        model_path = sys.argv[1]
    else:
        model_path = MODEL_PATH
    
    if len(sys.argv) >= 3:
        feature_dim = int(sys.argv[2])
    else:
        feature_dim = 84  # 默认84维（与训练v2数据一致）
    
    predictor = GesturePredictor(model_path, feature_dim)
    predictor.run()
