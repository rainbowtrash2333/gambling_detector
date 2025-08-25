#!/usr/bin/env python3
"""
CNN赌博网站检测器
使用PyTorch实现的CNN二分类模型，用于识别网站截图是否为赌博网站
- 训练数据: scan_results/train/0(非赌博) 和 scan_results/train/1(赌博)
- 测试数据: scan_results/screenshots/
- 输出: 将检测到的赌博网站截图复制到 scan_results/test/
- 支持GPU加速选项
"""

import os
import sys
import argparse
import time
import shutil
from pathlib import Path
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GamblingDataset(Dataset):
    """赌博网站数据集"""
    
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        Args:
            root_dir: 数据根目录
            transform: 图像预处理
            mode: 'train' 或 'predict'
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []
        
        if mode == 'train':
            self._load_training_data()
        else:
            self._load_prediction_data()
    
    def _load_training_data(self):
        """加载训练数据"""
        # 加载非赌博网站 (标签0)
        non_gambling_dir = self.root_dir / "0"
        if non_gambling_dir.exists():
            for img_path in non_gambling_dir.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(0)
        
        # 加载赌博网站 (标签1) 
        gambling_dir = self.root_dir / "1"
        if gambling_dir.exists():
            for img_path in gambling_dir.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(1)
        
        logger.info(f"加载训练数据: 非赌博={len([l for l in self.labels if l == 0])}, "
                   f"赌博={len([l for l in self.labels if l == 1])}")
    
    def _load_prediction_data(self):
        """加载预测数据"""
        for img_path in self.root_dir.glob("*.png"):
            self.images.append(str(img_path))
            self.labels.append(-1)  # 预测模式不需要标签
        
        logger.info(f"加载预测数据: {len(self.images)} 张图片")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            if self.mode == 'train':
                return image, self.labels[idx]
            else:
                return image, img_path  # 预测模式返回路径
                
        except Exception as e:
            logger.error(f"加载图片失败 {img_path}: {e}")
            # 返回黑色图片作为占位符
            dummy_image = torch.zeros(3, 224, 224)
            if self.mode == 'train':
                return dummy_image, self.labels[idx]
            else:
                return dummy_image, img_path

class GamblingCNN(nn.Module):
    """CNN赌博网站检测模型"""
    
    def __init__(self, num_classes=2):
        super(GamblingCNN, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第二个卷积块
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第三个卷积块
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四个卷积块
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第五个卷积块
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class GamblingDetector:
    """赌博网站检测器"""
    
    def __init__(self, use_gpu=True, model_path=None):
        """
        Args:
            use_gpu: 是否使用GPU
            model_path: 预训练模型路径
        """
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 数据预处理
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 创建模型
        self.model = GamblingCNN(num_classes=2).to(self.device)
        
        # 加载预训练模型
        if model_path and os.path.exists(model_path):
            logger.info(f"加载预训练模型: {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 目录设置
        self.train_dir = Path("scan_results/train")
        self.screenshots_dir = Path("scan_results/screenshots")
        self.test_dir = Path("scan_results/test")
        self.models_dir = Path("scan_results/models")
        self.model_save_path = self.models_dir / "gambling_detector_model.pth"
        
        # 创建输出目录
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, epochs=50, batch_size=32, learning_rate=0.001):
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 检查训练数据
        if not self.train_dir.exists():
            raise FileNotFoundError(f"训练数据目录不存在: {self.train_dir}")
        
        # 创建数据集
        train_dataset = GamblingDataset(
            self.train_dir, 
            transform=self.train_transform, 
            mode='train'
        )
        
        if len(train_dataset) == 0:
            raise ValueError("训练数据集为空")
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # 设置损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        
        # 训练循环
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # 显示进度
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                               f'Loss: {loss.item():.4f}')
            
            # 计算平均损失和准确率
            avg_loss = total_loss / len(train_loader)
            accuracy = 100.0 * correct / total
            
            logger.info(f'Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                
                # 保存模型状态字典
                torch.save(self.model.state_dict(), self.model_save_path)
                
                # 保存完整模型信息
                model_info = {
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch + 1,
                    'loss': best_loss,
                    'accuracy': accuracy,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_config': {
                        'num_classes': 2,
                        'input_size': (224, 224),
                        'batch_size': batch_size,
                        'learning_rate': learning_rate
                    },
                    'training_info': {
                        'total_epochs': epochs,
                        'best_epoch': epoch + 1,
                        'train_samples': len(train_dataset),
                        'device': str(self.device)
                    }
                }
                
                # 保存详细模型信息
                detailed_model_path = self.models_dir / f"gambling_detector_detailed_epoch{epoch+1}.pth"
                torch.save(model_info, detailed_model_path)
                
                logger.info(f"保存最佳模型: {self.model_save_path}")
                logger.info(f"保存详细信息: {detailed_model_path}")
            
            # 更新学习率
            scheduler.step()
        
        logger.info("训练完成!")
        
        # 保存最终模型和训练报告
        final_model_path = self.models_dir / "gambling_detector_final.pth"
        torch.save(self.model.state_dict(), final_model_path)
        
        # 创建训练报告
        training_report = {
            'training_completed': True,
            'total_epochs': epochs,
            'best_loss': best_loss,
            'final_accuracy': accuracy,
            'model_architecture': str(self.model),
            'training_samples': len(train_dataset),
            'device_used': str(self.device),
            'hyperparameters': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'optimizer': 'Adam',
                'scheduler': 'StepLR'
            },
            'model_files': {
                'best_model': str(self.model_save_path),
                'final_model': str(final_model_path),
                'detailed_models': str(self.models_dir / "gambling_detector_detailed_*.pth")
            }
        }
        
        # 保存训练报告
        report_path = self.models_dir / "training_report.json"
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(training_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存最终模型: {final_model_path}")
        logger.info(f"保存训练报告: {report_path}")
        logger.info(f"模型目录: {self.models_dir}")
    
    def predict(self, confidence_threshold=0.7):
        """对截图目录进行预测并复制赌博网站图片"""
        logger.info("开始预测...")
        
        # 检查模型是否存在
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_save_path}, 请先训练模型")
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        self.model.eval()
        
        # 检查截图目录
        if not self.screenshots_dir.exists():
            raise FileNotFoundError(f"截图目录不存在: {self.screenshots_dir}")
        
        # 创建预测数据集
        predict_dataset = GamblingDataset(
            self.screenshots_dir,
            transform=self.test_transform,
            mode='predict'
        )
        
        if len(predict_dataset) == 0:
            logger.warning("截图目录中没有图片")
            return
        
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # 预测结果
        gambling_images = []
        total_images = 0
        
        with torch.no_grad():
            for images, img_paths in predict_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                
                # 获取赌博网站的概率 (类别1)
                gambling_probs = probabilities[:, 1].cpu().numpy()
                
                for i, (prob, img_path) in enumerate(zip(gambling_probs, img_paths)):
                    total_images += 1
                    
                    # 如果概率超过阈值，认为是赌博网站
                    if prob >= confidence_threshold:
                        gambling_images.append((img_path, prob))
                        logger.info(f"检测到赌博网站: {Path(img_path).name} (置信度: {prob:.3f})")
        
        # 复制赌博网站图片到test目录
        if gambling_images:
            logger.info(f"共检测到 {len(gambling_images)} 个赌博网站，正在复制到 {self.test_dir}")
            
            for img_path, confidence in gambling_images:
                src_path = Path(img_path)
                dst_path = self.test_dir / f"{src_path.stem}_conf{confidence:.3f}{src_path.suffix}"
                
                try:
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"已复制: {src_path.name} -> {dst_path.name}")
                except Exception as e:
                    logger.error(f"复制文件失败 {src_path.name}: {e}")
        
        # 显示统计结果
        logger.info("=" * 60)
        logger.info("预测完成 - 统计报告")
        logger.info("=" * 60)
        logger.info(f"总图片数: {total_images}")
        logger.info(f"检测到赌博网站: {len(gambling_images)}")
        logger.info(f"检测率: {len(gambling_images)/total_images*100:.1f}%")
        logger.info(f"置信度阈值: {confidence_threshold}")
        logger.info(f"输出目录: {self.test_dir}")
        logger.info("=" * 60)
    
    def list_saved_models(self):
        """列出已保存的模型"""
        logger.info("已保存的模型:")
        logger.info("-" * 40)
        
        if not self.models_dir.exists():
            logger.info("模型目录不存在")
            return
        
        model_files = list(self.models_dir.glob("*.pth"))
        if not model_files:
            logger.info("没有找到模型文件")
            return
        
        for model_file in sorted(model_files):
            file_size = model_file.stat().st_size / (1024 * 1024)  # MB
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            logger.info(f"  {model_file.name} ({file_size:.1f}MB) - {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示训练报告
        report_path = self.models_dir / "training_report.json"
        if report_path.exists():
            logger.info(f"\n训练报告: {report_path}")
            try:
                import json
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                logger.info(f"  最佳损失: {report.get('best_loss', 'N/A')}")
                logger.info(f"  最终准确率: {report.get('final_accuracy', 'N/A')}%")
                logger.info(f"  训练样本数: {report.get('training_samples', 'N/A')}")
            except Exception as e:
                logger.warning(f"读取训练报告失败: {e}")
        
        logger.info("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='CNN赌博网站检测器')
    parser.add_argument('--mode', choices=['train', 'predict', 'both', 'list'], default='both',
                       help='运行模式: train(训练), predict(预测), both(训练+预测), list(列出模型)')
    parser.add_argument('--gpu', action='store_true', help='启用GPU加速')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--confidence', type=float, default=0.7, help='预测置信度阈值')
    parser.add_argument('--model_path', help='预训练模型路径')
    
    args = parser.parse_args()
    
    print("CNN赌博网站检测器")
    print("=" * 50)
    print(f"运行模式: {args.mode}")
    
    if args.mode != 'list':
        print(f"GPU加速: {'启用' if args.gpu else '禁用'}")
        
        # 检查PyTorch和CUDA
        print(f"PyTorch版本: {torch.__version__}")
        if args.gpu:
            if torch.cuda.is_available():
                print(f"CUDA版本: {torch.version.cuda}")
                print(f"GPU设备: {torch.cuda.get_device_name(0)}")
            else:
                print("警告: CUDA不可用，将使用CPU")
                args.gpu = False
    
    print("-" * 50)
    
    try:
        # 创建检测器
        detector = GamblingDetector(use_gpu=args.gpu, model_path=args.model_path)
        
        # 列出模型信息
        if args.mode == 'list':
            detector.list_saved_models()
            return
        
        # 执行训练
        if args.mode in ['train', 'both']:
            start_time = time.time()
            detector.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate
            )
            train_time = time.time() - start_time
            print(f"训练耗时: {train_time:.2f} 秒")
        
        # 执行预测
        if args.mode in ['predict', 'both']:
            start_time = time.time()
            detector.predict(confidence_threshold=args.confidence)
            predict_time = time.time() - start_time
            print(f"预测耗时: {predict_time:.2f} 秒")
        
        # 显示模型信息
        if args.mode in ['train', 'both']:
            print("\n" + "="*50)
            detector.list_saved_models()
        
        print("程序执行完成!")
        
    except FileNotFoundError as e:
        print(f"文件错误: {e}")
        print("请确保训练数据目录结构正确:")
        print("  scan_results/train/0/  (非赌博网站图片)")
        print("  scan_results/train/1/  (赌博网站图片)")
        print("  scan_results/screenshots/  (待检测图片)")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()