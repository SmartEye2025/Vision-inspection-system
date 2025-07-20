"""
LSTM训练模块：用于原子动作识别和异常行为检测
基于YOLO姿态估计提取的时序特征进行训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict
import argparse

class PoseSequenceDataset(Dataset):
    """
    姿态序列数据集类
    """
    def __init__(self, data_dir, sequence_length=30, stride=10):
        """
        Args:
            data_dir: pose_features目录路径
            sequence_length: LSTM输入序列长度
            stride: 滑动窗口步长
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        
        # 加载动作标签映射
        self.action_to_label, self.label_to_action = self._create_label_mapping()
        
        # 加载所有数据
        self.sequences, self.labels = self._load_data()
        
        print(f"数据集加载完成:")
        print(f"  - 总序列数: {len(self.sequences)}")
        print(f"  - 动作类别: {len(self.action_to_label)}")
        print(f"  - 序列长度: {self.sequence_length}")
        print(f"  - 特征维度: {self.sequences[0].shape[1] if len(self.sequences) > 0 else 'N/A'}")
    
    def _create_label_mapping(self):
        """创建动作标签映射"""
        # 获取所有动作文件夹名称
        action_folders = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        action_folders.sort()  # 确保标签顺序一致
        
        action_to_label = {action: idx for idx, action in enumerate(action_folders)}
        label_to_action = {idx: action for action, idx in action_to_label.items()}
        
        # 保存标签映射
        mapping_file = self.data_dir / "label_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump({
                'action_to_label': action_to_label,
                'label_to_action': label_to_action,
                'num_classes': len(action_to_label)
            }, f, indent=2, ensure_ascii=False)
        
        print(f"动作标签映射已保存: {mapping_file}")
        print(f"动作类别: {list(action_to_label.keys())}")
        
        return action_to_label, label_to_action
    
    def _load_data(self):
        """加载所有数据"""
        sequences = []
        labels = []
        
        for action_folder in self.data_dir.iterdir():
            if not action_folder.is_dir():
                continue
                
            action_name = action_folder.name
            if action_name not in self.action_to_label:
                continue
                
            action_label = self.action_to_label[action_name]
            
            # 遍历每个人的数据
            for person_folder in action_folder.iterdir():
                if not (person_folder.is_dir() and person_folder.name.startswith('person_')):
                    continue
                
                # 加载LSTM特征
                lstm_features_file = person_folder / f"{action_name}_lstm_features.npy"
                if not lstm_features_file.exists():
                    continue
                
                try:
                    lstm_features = np.load(lstm_features_file)  # (T, 68)
                    
                    # 检查数据质量
                    if len(lstm_features) < self.sequence_length:
                        continue  # 序列太短，跳过
                    
                    # 滑动窗口提取序列
                    for start_idx in range(0, len(lstm_features) - self.sequence_length + 1, self.stride):
                        end_idx = start_idx + self.sequence_length
                        sequence = lstm_features[start_idx:end_idx]  # (sequence_length, 68)
                        
                        # 检查是否有无效数据
                        if np.isnan(sequence).any() or np.isinf(sequence).any():
                            continue
                        
                        sequences.append(sequence)
                        labels.append(action_label)
                        
                except Exception as e:
                    print(f"加载数据出错 {lstm_features_file}: {e}")
                    continue
        
        return sequences, labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])  # (sequence_length, 68)
        label = torch.LongTensor([self.labels[idx]])[0]
        return sequence, label

class ActionLSTM(nn.Module):
    """
    动作识别LSTM模型
    针对50个动作类别进行优化
    """
    def __init__(self, input_size=68, hidden_size=192, num_layers=2, num_classes=50, dropout=0.3):
        super(ActionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层 - 调整隐藏层大小适应50类分类
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # 使用双向LSTM提高性能
        )
        
        # 双向LSTM的输出是hidden_size * 2
        lstm_output_size = hidden_size * 2
        
        # 多层全连接层以更好地处理复杂分类
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        # 多层全连接层以更好地处理复杂分类
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # 激活函数和正则化
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)  # 减少后层的dropout
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 对于双向LSTM，使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # 多层全连接网络
        out = self.fc1(last_output)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out

class AnomalyDetector:
    """
    异常行为检测器
    基于原子动作序列检测异常模式
    """
    def __init__(self, model_path, label_mapping_path):
        """
        Args:
            model_path: 训练好的模型路径
            label_mapping_path: 标签映射文件路径
        """
        # 加载标签映射
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        self.label_to_action = mapping_data['label_to_action']
        self.num_classes = mapping_data['num_classes']
        
        # 加载模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ActionLSTM(num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device
        
        # 定义异常规则
        self.anomaly_rules = self._define_anomaly_rules()
        
        print(f"异常检测器初始化完成")
        print(f"  - 动作类别数: {self.num_classes}")
        print(f"  - 异常规则数: {len(self.anomaly_rules)}")
    
    def _define_anomaly_rules(self):
        """定义异常行为规则"""
        # 这里定义异常的动作序列模式
        anomaly_rules = [
            # 规则格式: (前一个动作, 当前动作, 异常描述)
            ("focus", "run", "专注状态下突然奔跑 - 可能的逃离行为"),
            ("focus", "dance", "专注状态下突然跳舞 - 异常行为"),
            ("walk", "curlup", "行走中突然蜷缩 - 可能受伤或恐惧"),
            ("standup", "curlup", "刚站起又立即蜷缩 - 异常行为"),
            ("run", "curlup", "奔跑后突然蜷缩 - 可能跌倒或受伤"),
        ]
        
        return anomaly_rules
    
    def detect_anomaly_in_sequence(self, action_sequence):
        """
        检测动作序列中的异常
        
        Args:
            action_sequence: 动作序列，如 ["focus", "focus", "run", "run"]
            
        Returns:
            list: 检测到的异常情况
        """
        anomalies = []
        
        for i in range(1, len(action_sequence)):
            prev_action = action_sequence[i-1]
            curr_action = action_sequence[i]
            
            # 检查是否匹配异常规则
            for prev_rule, curr_rule, description in self.anomaly_rules:
                if prev_action.startswith(prev_rule) and curr_action.startswith(curr_rule):
                    anomalies.append({
                        'position': i,
                        'from_action': prev_action,
                        'to_action': curr_action,
                        'description': description,
                        'rule': f"{prev_rule} -> {curr_rule}"
                    })
        
        return anomalies
    
    def predict_action(self, pose_sequence):
        """
        预测单个姿态序列的动作
        
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
            
            predicted_action = self.label_to_action[str(predicted_label)]
            
        return predicted_action, confidence

def train_model(data_dir, model_save_path, epochs=80, batch_size=32, learning_rate=0.0008):
    """
    训练LSTM模型 - 针对50类动作优化参数，GPU加速版本
    """
    print("开始训练LSTM模型...")
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"🚀 检测到CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        device = torch.device('cuda')
        # 优化GPU内存使用
        torch.backends.cudnn.benchmark = True
    else:
        print("⚠️  未检测到CUDA GPU，使用CPU训练")
        device = torch.device('cpu')
    
    # 加载数据集
    dataset = PoseSequenceDataset(data_dir)
    
    if len(dataset) == 0:
        print("❌ 没有找到有效的训练数据!")
        return None
    
    # 分割训练集和测试集 (调整为85%训练，15%测试)
    train_size = int(0.85 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器 - 针对GPU优化batch_size和num_workers
    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    # 创建模型
    model = ActionLSTM(num_classes=len(dataset.action_to_label))
    model.to(device)
    
    # GPU优化：启用混合精度训练（如果支持）
    use_amp = device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 7
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("🔥 启用自动混合精度(AMP)训练以提升GPU性能")
    
    # 定义损失函数和优化器 - 使用标签平滑和权重衰减
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑有助于泛化
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # 使用AdamW
    
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 早停机制 - 减少patience因为类别数减少
    best_accuracy = 0
    patience = 15
    patience_counter = 0
    
    # 训练历史
    train_losses = []
    test_accuracies = []
    
    print(f"使用设备: {device}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"动作类别数: {len(dataset.action_to_label)}")
    print(f"批次大小: {batch_size}")
    
    # 开始训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 添加进度显示
        from tqdm import tqdm
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        
        for batch_idx, (sequences, labels) in enumerate(train_pbar):
            sequences, labels = sequences.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度训练
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_pbar.close()
        
        # 计算测试准确率
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            test_pbar = tqdm(test_loader, desc='Testing', leave=False)
            for sequences, labels in test_pbar:
                sequences, labels = sequences.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(sequences)
                else:
                    outputs = model(sequences)
                    
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_pbar.close()
        
        test_accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        test_accuracies.append(test_accuracy)
        
        # 早停检查
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path.replace('.pth', '_best.pth'))
        else:
            patience_counter += 1
        
        # 显示GPU内存使用情况
        if device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Best: {best_accuracy:.2f}%, GPU: {gpu_memory:.1f}GB')
        else:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Best: {best_accuracy:.2f}%')
        
        # 早停
        if patience_counter >= patience:
            print(f"早停触发! 在第{epoch+1}轮停止训练")
            break
        
        scheduler.step()
        
        # GPU内存清理
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 保存模型
    torch.save(model.state_dict(), model_save_path)
    print(f"✓ 模型已保存: {model_save_path}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(model_save_path.replace('.pth', '_training_curves.png'))
    plt.show()
    
    return model

def evaluate_model(model_path, data_dir, label_mapping_path):
    """
    评估模型性能
    """
    print("开始评估模型...")
    
    # 加载数据和模型
    dataset = PoseSequenceDataset(data_dir)
    _, test_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActionLSTM(num_classes=len(dataset.action_to_label))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 预测
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predicted)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    action_names = [dataset.label_to_action[i] for i in range(len(dataset.action_to_label))]
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=action_names, yticklabels=action_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(model_path.replace('.pth', '_confusion_matrix.png'))
    plt.show()
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_predicted, target_names=action_names))

def main():
    parser = argparse.ArgumentParser(description='LSTM动作识别训练 - 50类动作GPU加速版本')
    parser.add_argument('--data_dir', default='pose_features', help='数据目录')
    parser.add_argument('--model_path', default='action_lstm_model.pth', help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小(GPU优化)')
    parser.add_argument('--learning_rate', type=float, default=0.0008, help='学习率')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    
    args = parser.parse_args()
    
    if args.evaluate:
        label_mapping_path = os.path.join(args.data_dir, 'label_mapping.json')
        evaluate_model(args.model_path, args.data_dir, label_mapping_path)
    else:
        train_model(args.data_dir, args.model_path, args.epochs, args.batch_size, args.learning_rate)

if __name__ == "__main__":
    main()
