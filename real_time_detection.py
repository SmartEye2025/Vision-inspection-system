"""
实时异常行为检测系统
基于摄像头输入，实时检测人体姿态并识别异常行为
"""

import cv2
import torch
import numpy as np
import json
from collections import deque
import time
import threading
from pathlib import Path
import argparse
import winsound  # Windows系统发声
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
from train_lstm import ActionLSTM, AnomalyDetector

class RealTimeAnomalyDetector:
    """
    实时异常行为检测系统
    """
    def __init__(self, model_path, label_mapping_path, yolo_model_path="yolo11n-pose.pt"):
        """
        初始化检测系统
        
        Args:
            model_path: LSTM模型路径
            label_mapping_path: 动作标签映射文件路径
            yolo_model_path: YOLO姿态估计模型路径
        """
        # 加载YOLO模型
        print("加载YOLO姿态估计模型...")
        self.yolo_model = YOLO(yolo_model_path)
        
        # 加载LSTM异常检测器
        print("加载LSTM异常检测模型...")
        self.anomaly_detector = AnomalyDetector(model_path, label_mapping_path)
        
        # 参数设置
        self.sequence_length = 30  # LSTM输入序列长度
        self.confidence_threshold = 0.5  # 动作识别置信度阈值
        self.anomaly_threshold = 0.7  # 异常检测阈值
        
        # 存储每个人的特征序列
        self.person_sequences = {}  # person_id -> deque of features
        self.person_actions = {}    # person_id -> deque of actions
        self.person_trackers = {}   # person_id -> last_seen_frame
        
        # 异常检测结果
        self.current_anomalies = []
        
        # 统计信息
        self.frame_count = 0
        self.detection_count = 0
        self.anomaly_count = 0
        
        print("实时异常检测系统初始化完成!")
    
    def _extract_pose_features(self, keypoints):
        """
        从YOLO关键点提取LSTM特征
        
        Args:
            keypoints: YOLO关键点 (17, 3) - (x, y, confidence)
            
        Returns:
            numpy.ndarray: LSTM特征 (68,) - 归一化的(x, y, dx, dy)特征
        """
        # 提取坐标并归一化
        coords = keypoints[:, :2]  # (17, 2) - (x, y)
        
        # 归一化坐标（假设图像大小1280x720）
        normalized_coords = coords.copy()
        normalized_coords[:, 0] /= 1280.0  # x坐标归一化
        normalized_coords[:, 1] /= 720.0   # y坐标归一化
        
        # 创建特征向量 (17, 4) -> (68,)
        # 注意：dx, dy需要在序列中计算，这里先设为0
        features = np.zeros((17, 4))
        features[:, :2] = normalized_coords  # x, y
        features[:, 2:] = 0  # dx, dy 暂时设为0，将在序列中计算
        
        return features.flatten()  # (68,)
    
    def _calculate_motion_features(self, current_features, prev_features):
        """
        计算运动特征 (dx, dy)
        
        Args:
            current_features: 当前帧特征 (68,)
            prev_features: 前一帧特征 (68,)
            
        Returns:
            numpy.ndarray: 包含运动信息的特征 (68,)
        """
        features = current_features.copy()
        features_2d = features.reshape(17, 4)
        prev_features_2d = prev_features.reshape(17, 4)
        
        # 计算dx, dy
        features_2d[:, 2] = features_2d[:, 0] - prev_features_2d[:, 0]  # dx
        features_2d[:, 3] = features_2d[:, 1] - prev_features_2d[:, 1]  # dy
        
        return features_2d.flatten()
    
    def _track_persons(self, detections, frame_idx):
        """
        简单的人员跟踪
        
        Args:
            detections: YOLO检测结果
            frame_idx: 当前帧索引
            
        Returns:
            dict: person_id -> keypoints mapping
        """
        current_persons = {}
        
        if detections and len(detections) > 0:
            result = detections[0]
            
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                
                for i, keypoints in enumerate(keypoints_data):
                    person_id = f"person_{i}"
                    current_persons[person_id] = keypoints
                    self.person_trackers[person_id] = frame_idx
        
        # 清理长时间未见的人员
        expired_persons = []
        for person_id, last_seen in self.person_trackers.items():
            if frame_idx - last_seen > 30:  # 30帧未见则清理
                expired_persons.append(person_id)
        
        for person_id in expired_persons:
            if person_id in self.person_sequences:
                del self.person_sequences[person_id]
            if person_id in self.person_actions:
                del self.person_actions[person_id]
            del self.person_trackers[person_id]
        
        return current_persons
    
    def _detect_anomalies(self, person_id):
        """
        检测特定人员的异常行为
        
        Args:
            person_id: 人员ID
            
        Returns:
            list: 检测到的异常列表
        """
        if person_id not in self.person_actions:
            return []
        
        action_sequence = list(self.person_actions[person_id])
        if len(action_sequence) < 2:
            return []
        
        return self.anomaly_detector.detect_anomaly_in_sequence(action_sequence)
    
    def _send_alert(self, anomaly_info):
        """
        发送异常警报
        
        Args:
            anomaly_info: 异常信息字典
        """
        # 声音警报
        try:
            winsound.Beep(1000, 500)  # 1000Hz, 500ms
        except:
            pass  # 忽略声音错误
        
        # 打印警报
        print(f"\n🚨 异常行为检测!")
        print(f"  人员: {anomaly_info['person_id']}")
        print(f"  异常: {anomaly_info['description']}")
        print(f"  动作转换: {anomaly_info['from_action']} -> {anomaly_info['to_action']}")
        print(f"  检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # TODO: 这里可以添加其他警报方式
        # - 发送邮件
        # - 发送短信
        # - 写入日志文件
        # - 发送到监控中心
    
    def process_frame(self, frame):
        """
        处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            numpy.ndarray: 处理后的图像（带标注）
        """
        self.frame_count += 1
        
        # YOLO姿态检测
        detections = self.yolo_model(frame, verbose=False)
        
        # 人员跟踪
        current_persons = self._track_persons(detections, self.frame_count)
        
        # 处理每个人的数据
        for person_id, keypoints in current_persons.items():
            # 提取特征
            pose_features = self._extract_pose_features(keypoints)
            
            # 初始化序列存储
            if person_id not in self.person_sequences:
                self.person_sequences[person_id] = deque(maxlen=self.sequence_length)
                self.person_actions[person_id] = deque(maxlen=10)  # 保存最近10个动作
            
            # 计算运动特征
            if len(self.person_sequences[person_id]) > 0:
                prev_features = self.person_sequences[person_id][-1]
                pose_features = self._calculate_motion_features(pose_features, prev_features)
            
            # 添加到序列
            self.person_sequences[person_id].append(pose_features)
            
            # 当序列足够长时进行动作识别
            if len(self.person_sequences[person_id]) == self.sequence_length:
                sequence = np.array(list(self.person_sequences[person_id]))  # (30, 68)
                
                # 预测动作
                predicted_action, confidence = self.anomaly_detector.predict_action(sequence)
                
                if confidence > self.confidence_threshold:
                    self.person_actions[person_id].append(predicted_action)
                    self.detection_count += 1
                    
                    # 检测异常
                    anomalies = self._detect_anomalies(person_id)
                    
                    for anomaly in anomalies:
                        # 发送警报
                        anomaly_info = {
                            'person_id': person_id,
                            'frame_idx': self.frame_count,
                            **anomaly
                        }
                        self._send_alert(anomaly_info)
                        self.current_anomalies.append(anomaly_info)
                        self.anomaly_count += 1
        
        # 绘制检测结果
        annotated_frame = self._draw_annotations(frame, detections, current_persons)
        
        return annotated_frame
    
    def _draw_annotations(self, frame, detections, current_persons):
        """
        在图像上绘制检测结果和异常警报
        """
        annotated_frame = frame.copy()
        
        # 使用YOLO的原生绘制
        if detections and len(detections) > 0:
            annotated_frame = detections[0].plot()
        
        # 绘制额外信息
        height, width = annotated_frame.shape[:2]
        
        # 绘制统计信息
        info_text = [
            f"Frame: {self.frame_count}",
            f"Persons: {len(current_persons)}",
            f"Detections: {self.detection_count}",
            f"Anomalies: {self.anomaly_count}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制动作信息
        y_offset = 150
        for person_id in current_persons.keys():
            if person_id in self.person_actions and len(self.person_actions[person_id]) > 0:
                recent_actions = list(self.person_actions[person_id])[-3:]  # 最近3个动作
                action_text = f"{person_id}: {' -> '.join(recent_actions)}"
                cv2.putText(annotated_frame, action_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_offset += 25
        
        # 绘制异常警报
        if self.current_anomalies:
            # 显示最近的异常
            recent_anomaly = self.current_anomalies[-1]
            if self.frame_count - recent_anomaly['frame_idx'] < 60:  # 2秒内的异常
                warning_text = f"ANOMALY: {recent_anomaly['description']}"
                cv2.putText(annotated_frame, warning_text, (10, height - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                # 绘制警报边框
                cv2.rectangle(annotated_frame, (5, 5), (width-5, height-5), (0, 0, 255), 5)
        
        return annotated_frame
    
    def run_camera(self, camera_index=0):
        """
        运行摄像头实时检测
        
        Args:
            camera_index: 摄像头索引
        """
        print(f"启动摄像头检测 (camera {camera_index})...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {camera_index}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("实时异常检测已启动!")
        print("按 'q' 键退出")
        
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("错误: 无法读取摄像头帧")
                    break
                
                # 处理帧
                annotated_frame = self.process_frame(frame)
                
                # 计算FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = fps_counter / elapsed_time
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                               (annotated_frame.shape[1] - 150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow('Real-time Anomaly Detection', annotated_frame)
                
                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
        print(f"检测完成! 处理了 {self.frame_count} 帧，检测到 {self.anomaly_count} 个异常")
    
    def run_video(self, video_path, output_path=None):
        """
        运行视频文件检测
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
        """
        print(f"处理视频文件: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
        
        # 设置输出视频
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                annotated_frame = self.process_frame(frame)
                
                # 保存到输出视频
                if writer:
                    writer.write(annotated_frame)
                
                # 显示进度
                frame_idx += 1
                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    print(f"进度: {progress:.1f}% ({frame_idx}/{total_frames})")
                
                # 可选：显示实时预览
                # cv2.imshow('Video Processing', annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                    
        except KeyboardInterrupt:
            print("\n处理被用户中断")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
        print(f"视频处理完成! 检测到 {self.anomaly_count} 个异常")
        if output_path:
            print(f"结果已保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='实时异常行为检测系统')
    parser.add_argument('--model_path', required=True, help='LSTM模型路径')
    parser.add_argument('--label_mapping', help='标签映射文件路径')
    parser.add_argument('--yolo_model', default='yolo11n-pose.pt', help='YOLO模型路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--video', help='输入视频文件路径')
    parser.add_argument('--output', help='输出视频文件路径')
    
    args = parser.parse_args()
    
    # 自动找到标签映射文件
    if not args.label_mapping:
        model_dir = Path(args.model_path).parent
        label_mapping_path = model_dir / 'label_mapping.json'
        if not label_mapping_path.exists():
            label_mapping_path = 'pose_features/label_mapping.json'
        args.label_mapping = str(label_mapping_path)
    
    # 创建检测系统
    detector = RealTimeAnomalyDetector(
        args.model_path, 
        args.label_mapping, 
        args.yolo_model
    )
    
    # 运行检测
    if args.video:
        detector.run_video(args.video, args.output)
    else:
        detector.run_camera(args.camera)

if __name__ == "__main__":
    main()
