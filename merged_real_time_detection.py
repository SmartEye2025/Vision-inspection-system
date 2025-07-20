#!/usr/bin/env python3
"""
合并版异常行为检测器
基于8个主动作类别的实时异常检测
"""

import torch
import torch.nn as nn
import json
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from pathlib import Path
import time
import winsound
import argparse
from train_merged_lstm import MergedActionLSTM, extract_base_action

class MergedAnomalyDetector:
    """
    合并版异常行为检测器
    基于8个主动作类别检测异常模式
    """
    def __init__(self, model_path, label_mapping_path):
        """
        Args:
            model_path: 训练好的合并版模型路径
            label_mapping_path: 合并版标签映射文件路径
        """
        # 加载标签映射
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        self.label_to_action = {int(k): v for k, v in mapping_data['label_to_action'].items()}
        self.action_to_label = mapping_data['action_to_label']
        self.num_classes = mapping_data['num_classes']
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MergedActionLSTM(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device
        
        # 定义合并版异常规则
        self.anomaly_rules = self._define_merged_anomaly_rules()
        
        print(f"合并版异常检测器初始化完成")
        print(f"  - 主动作类别数: {self.num_classes}")
        print(f"  - 主动作类别: {list(self.action_to_label.keys())}")
        print(f"  - 异常规则数: {len(self.anomaly_rules)}")
    
    def _define_merged_anomaly_rules(self):
        """定义基于主动作的异常行为规则"""
        # 基于8个主动作类别的异常规则
        anomaly_rules = [
            # 规则格式: (前一个动作, 当前动作, 异常描述, 严重度)
            ("focus", "run", "专注状态下突然奔跑 - 可能的逃离行为", "high"),
            ("focus", "dance", "专注状态下突然跳舞 - 异常行为", "medium"),
            ("walk", "curlup", "行走中突然蜷缩 - 可能受伤或恐惧", "high"),
            ("standup", "curlup", "刚站起又立即蜷缩 - 异常行为", "medium"), 
            ("run", "curlup", "奔跑后突然蜷缩 - 可能跌倒或受伤", "high"),
            ("handup", "run", "举手后突然奔跑 - 可能求救后逃跑", "high"),
            ("turn-head", "run", "转头观察后奔跑 - 可能发现危险", "medium"),
            ("dance", "curlup", "跳舞时突然蜷缩 - 异常状态变化", "medium"),
        ]
        
        return anomaly_rules
    
    def detect_anomaly_in_sequence(self, action_sequence):
        """
        检测主动作序列中的异常
        
        Args:
            action_sequence: 主动作序列，如 ["focus", "focus", "run", "run"]
            
        Returns:
            list: 检测到的异常情况
        """
        anomalies = []
        
        for i in range(1, len(action_sequence)):
            prev_action = action_sequence[i-1]
            curr_action = action_sequence[i]
            
            # 检查是否匹配异常规则
            for prev_rule, curr_rule, description, severity in self.anomaly_rules:
                if prev_action == prev_rule and curr_action == curr_rule:
                    anomalies.append({
                        'position': i,
                        'from_action': prev_action,
                        'to_action': curr_action,
                        'description': description,
                        'severity': severity,
                        'rule': f"{prev_rule} -> {curr_rule}",
                        'timestamp': time.time()
                    })
        
        return anomalies
    
    def predict_action(self, pose_sequence):
        """
        预测单个姿态序列的主动作
        
        Args:
            pose_sequence: 姿态特征序列 (sequence_length, 68)
            
        Returns:
            tuple: (predicted_action, confidence)
        """
        with torch.no_grad():
            # 转换为tensor并添加batch维度
            sequence_tensor = torch.FloatTensor(pose_sequence).unsqueeze(0).to(self.device)
            
            # 预测
            outputs = self.model(sequence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            predicted_label = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_label].item()
            
            predicted_action = self.label_to_action[predicted_label]
            
        return predicted_action, confidence

class MergedRealTimeAnomalyDetector:
    """
    合并版实时异常检测系统
    支持摄像头和视频文件输入
    """
    def __init__(self, lstm_model_path, pose_model_path='yolo11n-pose.pt', sequence_length=30):
        self.yolo_model = YOLO(pose_model_path)
        self.sequence_length = sequence_length
        
        # 加载异常检测器
        label_mapping_path = Path('pose_features') / 'merged_label_mapping.json'
        self.anomaly_detector = MergedAnomalyDetector(lstm_model_path, label_mapping_path)
        
        # 人物轨迹管理
        self.person_tracks = {}  # track_id -> pose_sequence
        self.person_actions = {}  # track_id -> recent_actions
        self.action_history = {}  # track_id -> action_history
        
        # 异常警报管理
        self.recent_anomalies = deque(maxlen=10)
        self.last_alert_time = 0
        self.alert_cooldown = 5.0  # 秒
        
        print("合并版实时异常检测系统初始化完成")
        print(f"序列长度: {sequence_length}")
        print(f"主动作类别: {list(self.anomaly_detector.action_to_label.keys())}")
    
    def _normalize_keypoints(self, keypoints, img_width, img_height):
        """标准化关键点坐标到0-1范围"""
        normalized = keypoints.copy()
        normalized[:, 0] = keypoints[:, 0] / img_width  # x坐标标准化
        normalized[:, 1] = keypoints[:, 1] / img_height  # y坐标标准化
        return normalized
    
    def _calculate_enhanced_features(self, keypoints_sequence):
        """计算增强的姿态特征"""
        if len(keypoints_sequence) < 2:
            return None
            
        enhanced_features = []
        
        for i in range(len(keypoints_sequence)):
            current_kpts = keypoints_sequence[i]  # (17, 2)
            
            # 计算运动特征（与前一帧的差值）
            if i > 0:
                prev_kpts = keypoints_sequence[i-1]
                motion = current_kpts - prev_kpts  # (17, 2)
            else:
                motion = np.zeros_like(current_kpts)
            
            # 组合特征：(x, y, dx, dy) for each keypoint
            frame_features = np.concatenate([current_kpts, motion], axis=1)  # (17, 4)
            enhanced_features.append(frame_features)
        
        enhanced_features = np.array(enhanced_features)  # (T, 17, 4)
        
        # 转换为LSTM输入格式 (T, 68)
        lstm_features = enhanced_features.reshape(enhanced_features.shape[0], -1)
        
        return lstm_features
    
    def _trigger_alert(self, anomaly, track_id):
        """触发异常警报"""
        current_time = time.time()
        
        # 检查警报冷却时间
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        self.recent_anomalies.append(anomaly)
        
        # 根据严重度选择不同的警报
        if anomaly['severity'] == 'high':
            # 高危险：连续警报声
            for _ in range(3):
                winsound.Beep(1000, 200)
                time.sleep(0.1)
            print(f"🚨 高危警报! 人员{track_id}: {anomaly['description']}")
        else:
            # 中等危险：单次警报
            winsound.Beep(800, 500)
            print(f"⚠️  异常警报! 人员{track_id}: {anomaly['description']}")
    
    def process_frame(self, frame):
        """处理单帧图像"""
        img_height, img_width = frame.shape[:2]
        
        # YOLO姿态检测
        results = self.yolo_model.track(frame, persist=True, verbose=False)
        
        if results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            keypoints = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
            confidences = results[0].keypoints.conf.cpu().numpy()  # (N, 17)
            
            for i, track_id in enumerate(track_ids):
                kpts = keypoints[i]  # (17, 2)
                conf = confidences[i]  # (17,)
                
                # 过滤低置信度关键点
                valid_mask = conf > 0.3
                if np.sum(valid_mask) < 10:  # 至少需要10个有效关键点
                    continue
                
                # 标准化关键点
                normalized_kpts = self._normalize_keypoints(kpts, img_width, img_height)
                
                # 更新人物轨迹
                if track_id not in self.person_tracks:
                    self.person_tracks[track_id] = deque(maxlen=self.sequence_length)
                    self.person_actions[track_id] = deque(maxlen=10)
                    self.action_history[track_id] = []
                
                self.person_tracks[track_id].append(normalized_kpts)
                
                # 当轨迹足够长时进行动作预测
                if len(self.person_tracks[track_id]) == self.sequence_length:
                    keypoints_seq = list(self.person_tracks[track_id])
                    lstm_features = self._calculate_enhanced_features(keypoints_seq)
                    
                    if lstm_features is not None and len(lstm_features) == self.sequence_length:
                        # 预测主动作
                        predicted_action, confidence = self.anomaly_detector.predict_action(lstm_features)
                        
                        # 更新动作历史
                        if confidence > 0.6:  # 只记录高置信度的预测
                            self.person_actions[track_id].append(predicted_action)
                            self.action_history[track_id].append({
                                'action': predicted_action,
                                'confidence': confidence,
                                'timestamp': time.time()
                            })
                            
                            # 异常检测
                            if len(self.person_actions[track_id]) >= 2:
                                recent_actions = list(self.person_actions[track_id])[-5:]  # 最近5个动作
                                anomalies = self.anomaly_detector.detect_anomaly_in_sequence(recent_actions)
                                
                                # 处理检测到的异常
                                for anomaly in anomalies:
                                    self._trigger_alert(anomaly, track_id)
                        
                        # 在图像上绘制信息
                        x, y, w, h = boxes[i]
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        
                        # 绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 绘制动作信息
                        if confidence > 0.6:
                            text = f"ID:{track_id} {predicted_action} ({confidence:.2f})"
                            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # 如果最近有异常，用红色标记
                            if any(time.time() - a['timestamp'] < 3 for a in self.recent_anomalies):
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        return frame
    
    def run_camera(self, camera_index=0):
        """运行摄像头检测"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("开始摄像头异常检测... 按 'q' 退出")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 显示结果
                cv2.imshow('合并版实时异常检测', processed_frame)
                
                # 退出检查
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("检测被用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def run_video(self, video_path):
        """运行视频文件检测"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return
        
        print(f"开始视频异常检测: {video_path}")
        print("按 'q' 退出, 按空格暂停/继续")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("视频处理完成")
                        break
                    
                    # 处理帧
                    processed_frame = self.process_frame(frame)
                    
                    # 显示结果
                    cv2.imshow('合并版视频异常检测', processed_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print("视频已暂停" if paused else "视频继续播放")
                    
        except KeyboardInterrupt:
            print("检测被用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='合并版实时异常行为检测')
    parser.add_argument('--model', default='merged_action_lstm_model_best.pth', help='LSTM模型路径')
    parser.add_argument('--pose_model', default='yolo11n-pose.pt', help='YOLO姿态模型路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--video', type=str, help='视频文件路径')
    parser.add_argument('--sequence_length', type=int, default=30, help='LSTM序列长度')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = MergedRealTimeAnomalyDetector(
        lstm_model_path=args.model,
        pose_model_path=args.pose_model,
        sequence_length=args.sequence_length
    )
    
    # 运行检测
    if args.video:
        detector.run_video(args.video)
    else:
        detector.run_camera(args.camera)

if __name__ == "__main__":
    main()
