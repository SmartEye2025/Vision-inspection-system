import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
from pathlib import Path
import argparse
from tqdm import tqdm

class PoseFeatureExtractor:
    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        初始化姿态特征提取器
        
        Args:
            model_path: YOLO模型路径
        """
        self.model = YOLO(model_path)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def extract_pose_from_image(self, image_path, min_confidence=0.3):
        """
        从单张图片中提取所有人的姿态关键点
        
        Args:
            image_path: 图片路径
            min_confidence: 最小置信度阈值
            
        Returns:
            list: 每个人的关键点数据列表
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"警告: 无法读取图片 {image_path}")
            return []
        
        # 运行姿态估计
        results = self.model(frame, verbose=False)
        
        persons_data = []
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
            keypoints_conf = results[0].keypoints.conf.cpu().numpy()  # (N, 17)
            
            # 遍历每个检测到的人
            for person_idx in range(len(keypoints)):
                person_keypoints = keypoints[person_idx]  # (17, 2)
                person_conf = keypoints_conf[person_idx]  # (17,)
                
                # 过滤低置信度的关键点
                valid_keypoints = []
                for kp_idx in range(len(person_keypoints)):
                    x, y = person_keypoints[kp_idx]
                    conf = person_conf[kp_idx]
                    
                    if conf >= min_confidence:
                        valid_keypoints.append({
                            'keypoint_id': kp_idx,
                            'keypoint_name': self.keypoint_names[kp_idx],
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(conf)
                        })
                    else:
                        # 低置信度的点设置为0
                        valid_keypoints.append({
                            'keypoint_id': kp_idx,
                            'keypoint_name': self.keypoint_names[kp_idx],
                            'x': 0.0,
                            'y': 0.0,
                            'confidence': 0.0
                        })
                
                # 计算人体中心点（用于后续跟踪和分离）
                valid_points = [(kp['x'], kp['y']) for kp in valid_keypoints if kp['confidence'] > 0]
                if valid_points:
                    center_x = np.mean([p[0] for p in valid_points])
                    center_y = np.mean([p[1] for p in valid_points])
                else:
                    center_x = center_y = 0
                
                person_data = {
                    'person_id': person_idx,
                    'center': {'x': float(center_x), 'y': float(center_y)},
                    'keypoints': valid_keypoints,
                    'keypoints_array': person_keypoints.tolist(),  # 原始坐标数组
                    'confidence_array': person_conf.tolist()       # 置信度数组
                }
                
                persons_data.append(person_data)
        
        return persons_data
    
    def process_video_sequence(self, image_folder, output_folder, action_name):
        """
        处理一个视频序列（一个文件夹的图片）
        
        Args:
            image_folder: 图片文件夹路径
            output_folder: 输出文件夹路径
            action_name: 动作名称
        """
        # 获取所有图片文件并排序
        image_files = sorted([f for f in os.listdir(image_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            print(f"警告: 文件夹 {image_folder} 中没有找到图片文件")
            return
        
        print(f"处理动作: {action_name}, 图片数量: {len(image_files)}")
        
        # 存储所有帧的数据
        sequence_data = []
        person_trajectories = {}  # 用于跟踪不同的人
        
        for img_idx, image_file in enumerate(tqdm(image_files, desc=f"处理 {action_name}")):
            image_path = os.path.join(image_folder, image_file)
            frame_data = self.extract_pose_from_image(image_path)
            
            # 添加帧信息
            frame_info = {
                'frame_id': img_idx,
                'image_name': image_file,
                'persons': frame_data
            }
            sequence_data.append(frame_info)
            
            # 简单的人员跟踪（基于位置距离）
            if frame_data:
                self._track_persons(frame_data, person_trajectories, img_idx)
        
        # 保存完整序列数据
        self._save_sequence_data(sequence_data, person_trajectories, output_folder, action_name)
    
    def _track_persons(self, frame_data, person_trajectories, frame_idx, max_distance=100):
        """
        简单的人员跟踪算法
        """
        for person in frame_data:
            center = person['center']
            if center['x'] == 0 and center['y'] == 0:
                continue
                
            # 寻找最近的轨迹
            best_track_id = None
            min_distance = float('inf')
            
            for track_id, trajectory in person_trajectories.items():
                if len(trajectory) > 0:
                    last_center = trajectory[-1]['center']
                    distance = np.sqrt((center['x'] - last_center['x'])**2 + 
                                     (center['y'] - last_center['y'])**2)
                    if distance < min_distance and distance < max_distance:
                        min_distance = distance
                        best_track_id = track_id
            
            # 分配轨迹ID
            if best_track_id is not None:
                track_id = best_track_id
            else:
                # 创建新轨迹
                track_id = len(person_trajectories)
                person_trajectories[track_id] = []
            
            # 添加到轨迹
            person_copy = person.copy()
            person_copy['frame_id'] = frame_idx
            person_copy['track_id'] = track_id
            person_trajectories[track_id].append(person_copy)
    
    def _calculate_enhanced_features(self, keypoints_sequence):
        """
        计算增强特征：(x, y, dx, dy) + LSTM友好的格式
        
        Args:
            keypoints_sequence: 关键点序列 (T, 17, 2)
            
        Returns:
            tuple: (enhanced_features, lstm_features)
                - enhanced_features: (T, 17, 4) - (x, y, dx, dy)
                - lstm_features: (T, 68) - 展平后的特征，适合LSTM训练
        """
        keypoints_array = np.array(keypoints_sequence)  # (T, 17, 2)
        T, num_keypoints, _ = keypoints_array.shape
        
        # 初始化增强特征数组 (T, 17, 4)
        enhanced_features = np.zeros((T, num_keypoints, 4))
        
        # 坐标归一化（假设图像大小为1280x720）
        normalized_keypoints = keypoints_array.copy()
        normalized_keypoints[:, :, 0] /= 1280.0  # x坐标归一化
        normalized_keypoints[:, :, 1] /= 720.0   # y坐标归一化
        
        # 填充归一化的 x, y 坐标
        enhanced_features[:, :, :2] = normalized_keypoints  # x, y
        
        # 计算位移 dx, dy
        for t in range(1, T):
            # 当前帧与前一帧的位移
            dx = normalized_keypoints[t, :, 0] - normalized_keypoints[t-1, :, 0]  # x位移
            dy = normalized_keypoints[t, :, 1] - normalized_keypoints[t-1, :, 1]  # y位移
            
            enhanced_features[t, :, 2] = dx  # dx
            enhanced_features[t, :, 3] = dy  # dy
        
        # 第一帧的位移设为0（因为没有前一帧）
        enhanced_features[0, :, 2:] = 0
        
        # 创建LSTM友好的格式：展平为 (T, 68) - 17个关键点 × 4个特征
        lstm_features = enhanced_features.reshape(T, -1)
        
        return enhanced_features, lstm_features
    
    def _save_sequence_data(self, sequence_data, person_trajectories, output_folder, action_name):
        """
        保存序列数据
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 保存完整的序列数据
        sequence_file = os.path.join(output_folder, f"{action_name}_full_sequence.json")
        with open(sequence_file, 'w', encoding='utf-8') as f:
            json.dump(sequence_data, f, indent=2, ensure_ascii=False)
        
        # 为每个人单独保存轨迹数据
        for track_id, trajectory in person_trajectories.items():
            if len(trajectory) < 5:  # 过滤太短的轨迹
                continue
                
            # 提取关键点序列用于LSTM训练
            keypoints_sequence = []
            confidence_sequence = []
            
            for frame_data in trajectory:
                keypoints_sequence.append(frame_data['keypoints_array'])
                confidence_sequence.append(frame_data['confidence_array'])
            
            # 计算增强特征：(x, y, dx, dy) 和 LSTM格式
            enhanced_features, lstm_features = self._calculate_enhanced_features(keypoints_sequence)
            
            # 保存numpy格式（用于LSTM训练）
            person_folder = os.path.join(output_folder, f"person_{track_id}")
            os.makedirs(person_folder, exist_ok=True)
            
            # 保存原始关键点和增强特征
            np.save(os.path.join(person_folder, f"{action_name}_keypoints.npy"), 
                   np.array(keypoints_sequence))
            np.save(os.path.join(person_folder, f"{action_name}_enhanced_features.npy"), 
                   enhanced_features)
            np.save(os.path.join(person_folder, f"{action_name}_lstm_features.npy"), 
                   lstm_features)  # 新增：LSTM友好格式
            np.save(os.path.join(person_folder, f"{action_name}_confidence.npy"), 
                   np.array(confidence_sequence))
            
            # 保存轨迹元数据
            metadata = {
                'action_name': action_name,
                'track_id': track_id,
                'sequence_length': len(trajectory),
                'keypoints_shape': np.array(keypoints_sequence).shape,
                'enhanced_features_shape': enhanced_features.shape,
                'lstm_features_shape': lstm_features.shape,  # 新增
                'confidence_shape': np.array(confidence_sequence).shape,
                'feature_description': {
                    'enhanced_features': '(x, y, dx, dy) for each keypoint - shape (T, 17, 4)',
                    'lstm_features': 'flattened features for LSTM - shape (T, 68)',
                    'feature_dimensions': 4,  # x, y, dx, dy
                    'keypoints_count': 17
                }
            }
            
            with open(os.path.join(person_folder, f"{action_name}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✓ 保存完成: {action_name}, 检测到 {len(person_trajectories)} 个人的轨迹")
    
    def process_dataset(self, input_root, output_root, action_folders=None):
        """
        处理整个数据集
        
        Args:
            input_root: 输入数据集根目录
            output_root: 输出目录
            action_folders: 要处理的动作文件夹列表，None表示处理所有
        """
        input_path = Path(input_root)
        output_path = Path(output_root)
        
        if not input_path.exists():
            raise ValueError(f"输入目录不存在: {input_root}")
        
        # 获取所有动作文件夹
        if action_folders is None:
            action_folders = [d.name for d in input_path.iterdir() if d.is_dir()]
        
        print(f"开始处理数据集，共 {len(action_folders)} 个动作文件夹")
        print(f"输入目录: {input_root}")
        print(f"输出目录: {output_root}")
        print(f"动作列表: {action_folders}")
        
        for action_name in action_folders:
            action_input_folder = input_path / action_name
            action_output_folder = output_path / action_name
            
            if not action_input_folder.exists():
                print(f"警告: 动作文件夹不存在 {action_input_folder}")
                continue
            
            try:
                self.process_video_sequence(
                    str(action_input_folder), 
                    str(action_output_folder), 
                    action_name
                )
            except Exception as e:
                print(f"错误: 处理 {action_name} 时出现异常: {e}")
                continue
        
        print("✓ 数据集处理完成!")
        
        # 生成数据集统计信息
        self._generate_dataset_summary(output_path)
    
    def _generate_dataset_summary(self, output_root):
        """
        生成数据集统计信息
        """
        summary = {
            'total_actions': 0,
            'total_persons': 0,
            'actions': {}
        }
        
        for action_folder in Path(output_root).iterdir():
            if not action_folder.is_dir():
                continue
                
            action_name = action_folder.name
            person_folders = [d for d in action_folder.iterdir() if d.is_dir() and d.name.startswith('person_')]
            
            summary['actions'][action_name] = {
                'person_count': len(person_folders),
                'persons': []
            }
            
            for person_folder in person_folders:
                metadata_file = person_folder / f"{action_name}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        summary['actions'][action_name]['persons'].append(metadata)
            
            summary['total_actions'] += 1
            summary['total_persons'] += len(person_folders)
        
        # 保存统计信息
        with open(output_root / 'dataset_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 数据集统计: {summary['total_actions']} 个动作, {summary['total_persons']} 个人物轨迹")
        print("✓ 增强特征说明: 每个关键点包含 (x, y, dx, dy) 共4个特征维度")


def main():
    parser = argparse.ArgumentParser(description='提取姿态特征用于LSTM训练')
    parser.add_argument('--input', '-i', required=True, help='输入数据集目录')
    parser.add_argument('--output', '-o', required=True, help='输出目录')
    parser.add_argument('--model', '-m', default='yolo11n-pose.pt', help='YOLO模型路径')
    parser.add_argument('--actions', '-a', nargs='+', help='指定要处理的动作文件夹')
    
    args = parser.parse_args()
    
    # 创建特征提取器
    extractor = PoseFeatureExtractor(args.model)
    
    # 处理数据集
    extractor.process_dataset(args.input, args.output, args.actions)


if __name__ == "__main__":
    # 如果直接运行脚本，使用默认参数
    extractor = PoseFeatureExtractor('yolo11n-pose.pt')
    
    # 处理您的数据集
    input_directory = "cleaned_data"  # 您的输入目录
    output_directory = "pose_features"  # 输出目录
    
    extractor.process_dataset(input_directory, output_directory)
