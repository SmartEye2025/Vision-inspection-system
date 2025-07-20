#!/usr/bin/env python3
"""
合并版本的LSTM训练模块：将细分动作合并成主动作类别
从50个细分动作合并成8个主动作，大幅增加每类的样本数量
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict
import argparse

def extract_base_action(action_name):
    """提取动作的基本类别名称"""
    # 移除数字后缀
    base_action = re.sub(r'\d+$', '', action_name)
    
    # 特殊情况处理
    if base_action == 'turn-head':
        return 'turn-head'
    elif base_action == 'walk squatting':
        return 'walk'  # 将walk squatting归类为walk
    elif 'standup' in base_action and '问题' in action_name:
        return 'standup'  # 将有问题的standup也归为standup类
    else:
        return base_action

class MergedPoseSequenceDataset(Dataset):
    """
    合并版姿态序列数据集类
    将细分动作合并成主动作类别
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
        
        # 加载动作标签映射（合并版本）
        self.action_to_label, self.label_to_action = self._create_merged_label_mapping()
        
        # 加载所有数据
        self.sequences, self.labels = self._load_merged_data()
        
        print(f"合并后数据集加载完成:")
        print(f"  - 总序列数: {len(self.sequences)}")
        print(f"  - 主动作类别: {len(self.action_to_label)}")
        print(f"  - 序列长度: {self.sequence_length}")
        print(f"  - 特征维度: {self.sequences[0].shape[1] if len(self.sequences) > 0 else 'N/A'}")
    
    def _create_merged_label_mapping(self):
        """创建合并版动作标签映射"""
        # 获取所有动作文件夹名称
        action_folders = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        action_folders.sort()
        
        # 提取基本动作类别
        base_actions = set()
        for action in action_folders:
            base = extract_base_action(action)
            base_actions.add(base)
        
        base_actions = sorted(list(base_actions))
        
        action_to_label = {action: idx for idx, action in enumerate(base_actions)}
        label_to_action = {idx: action for action, idx in action_to_label.items()}
        
        # 保存标签映射
        mapping_file = self.data_dir / "merged_label_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump({
                'action_to_label': action_to_label,
                'label_to_action': label_to_action,
                'num_classes': len(action_to_label)
            }, f, indent=2, ensure_ascii=False)
        
        print(f"合并版动作标签映射已保存: {mapping_file}")
        print(f"主动作类别: {list(action_to_label.keys())}")
        
        return action_to_label, label_to_action
    
    def _load_merged_data(self):
        """加载所有数据（合并版本）"""
        sequences = []
        labels = []
        
        # 统计每个主动作的样本数
        main_action_counts = defaultdict(int)
        
        for action_folder in self.data_dir.iterdir():
            if not action_folder.is_dir():
                continue
                
            action_name = action_folder.name
            base_action = extract_base_action(action_name)
            
            if base_action not in self.action_to_label:
                continue
                
            action_label = self.action_to_label[base_action]
            
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
                        main_action_counts[base_action] += 1
                        
                except Exception as e:
                    print(f"加载数据出错 {lstm_features_file}: {e}")
                    continue
        
        # 打印每个主动作的样本统计
        print(f"\n=== 合并后样本分布 ===")
        for action, count in sorted(main_action_counts.items()):
            print(f"{action}: {count} 个序列")
        
        return sequences, labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])  # (sequence_length, 68)
        label = torch.LongTensor([self.labels[idx]])[0]
        return sequence, label

class MergedActionLSTM(nn.Module):
    """
    合并版动作识别LSTM模型
    针对8个主动作类别进行优化
    """
    def __init__(self, input_size=68, hidden_size=128, num_layers=2, num_classes=8, dropout=0.25):
        super(MergedActionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层 - 8类分类可以使用更轻量的模型
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
        
        # 简化的全连接层（8类分类不需要太复杂）
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # 激活函数和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 对于双向LSTM，使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        
        # 全连接网络
        out = self.fc1(last_output)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def train_merged_model(data_dir, model_save_path, epochs=60, batch_size=48, learning_rate=0.001):
    """
    训练合并版LSTM模型 - GPU加速版本，针对8类主动作优化
    """
    print("开始训练合并版LSTM模型...")
    
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
    dataset = MergedPoseSequenceDataset(data_dir)
    
    if len(dataset) == 0:
        print("❌ 没有找到有效的训练数据!")
        return None
    
    # 分割训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器 - GPU优化版本
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
    model = MergedActionLSTM(num_classes=len(dataset.action_to_label))
    model.to(device)
    
    # GPU优化：启用混合精度训练（如果支持）
    use_amp = device.type == 'cuda' and torch.cuda.get_device_capability(0)[0] >= 7
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    if use_amp:
        print("🔥 启用自动混合精度(AMP)训练以提升GPU性能")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)
    
    # 早停机制
    best_accuracy = 0
    patience = 15
    patience_counter = 0
    
    # 训练历史
    train_losses = []
    test_accuracies = []
    
    print(f"使用设备: {device}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"主动作类别数: {len(dataset.action_to_label)}")
    print(f"批次大小: {batch_size}")
    
    # 开始训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 添加进度显示
        try:
            from tqdm import tqdm
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        except ImportError:
            train_pbar = train_loader
        
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
            
            if hasattr(train_pbar, 'set_postfix'):
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        if hasattr(train_pbar, 'close'):
            train_pbar.close()
        
        # 计算测试准确率
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
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
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Best: {best_accuracy:.2f}%')
        
        # 调整学习率
        scheduler.step(test_accuracy)
        
        # 早停
        if patience_counter >= patience:
            print(f"早停触发! 在第{epoch+1}轮停止训练")
            break
    
    # 保存最终模型
    torch.save(model.state_dict(), model_save_path)
    print(f"✓ 模型已保存: {model_save_path}")
    print(f"✓ 最佳模型已保存: {model_save_path.replace('.pth', '_best.pth')}")
    
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

def evaluate_merged_model(model_path, data_dir):
    """
    评估合并版模型性能
    """
    print("开始评估合并版模型...")
    
    # 加载数据和模型
    dataset = MergedPoseSequenceDataset(data_dir)
    _, test_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MergedActionLSTM(num_classes=len(dataset.action_to_label))
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
    plt.figure(figsize=(8, 6))
    action_names = [dataset.label_to_action[i] for i in range(len(dataset.action_to_label))]
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=action_names, yticklabels=action_names)
    plt.title('Merged Actions Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(model_path.replace('.pth', '_confusion_matrix.png'))
    plt.show()
    
    # 打印分类报告
    print("\n合并版分类报告:")
    print(classification_report(all_labels, all_predicted, target_names=action_names))

def main():
    parser = argparse.ArgumentParser(description='合并版LSTM动作识别训练 - 8类主动作GPU加速版本')
    parser.add_argument('--data_dir', default='pose_features', help='数据目录')
    parser.add_argument('--model_path', default='merged_action_lstm_model.pth', help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=48, help='批次大小(GPU优化)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_merged_model(args.model_path, args.data_dir)
    else:
        train_merged_model(args.data_dir, args.model_path, args.epochs, args.batch_size, args.learning_rate)

if __name__ == "__main__":
    main()
