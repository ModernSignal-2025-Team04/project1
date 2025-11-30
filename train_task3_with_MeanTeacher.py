# mean_teacher_task3.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms.functional as TF
import random

from dataset_task3 import Task3AugDataset
from dataset_faz import FazSegDataset
from unet import UNet2D
from metrics import dice_loss, dice_coeff


class MeanTeacherTrainer:
    def __init__(self, model, ema_model, optimizer, device):
        self.model = model
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.device = device
        
        # 损失函数
        self.supervised_loss = nn.BCEWithLogitsLoss()
        self.consistency_loss = nn.MSELoss()
        
        # Mean Teacher 参数
        self.consistency_weight = 3.0
        self.ema_decay = 0.99
        self.consistency_rampup = 30
        
        self.global_step = 0
        
    def update_ema(self, alpha):
        """更新EMA教师模型"""
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    
    def consistency_weight_rampup(self, epoch):
        """一致性损失权重逐渐增加"""
        return self.consistency_weight * min(epoch / self.consistency_rampup, 1.0)
    
    def train_epoch(self, labeled_loader, unlabeled_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        self.ema_model.train()
        
        total_supervised_loss = 0
        total_consistency_loss = 0
        total_loss = 0
        
        # 同时遍历有标签和无标签数据
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # 以较长的数据加载器为准
        max_length = max(len(labeled_loader), len(unlabeled_loader))
        
        pbar = tqdm(range(max_length), desc=f"Epoch {epoch}")
        
        for i in pbar:
            # 获取有标签数据
            try:
                labeled_imgs, labeled_masks = next(labeled_iter)
                labeled_imgs = labeled_imgs.to(self.device)
                labeled_masks = labeled_masks.to(self.device)
                has_labeled = True
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_imgs, labeled_masks = next(labeled_iter)
                labeled_imgs = labeled_imgs.to(self.device)
                labeled_masks = labeled_masks.to(self.device)
                has_labeled = True
            
            # 获取无标签数据
            try:
                unlabeled_imgs, _ = next(unlabeled_iter)
                unlabeled_imgs = unlabeled_imgs.to(self.device)
                has_unlabeled = True
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_imgs, _ = next(unlabeled_iter)
                unlabeled_imgs = unlabeled_imgs.to(self.device)
                has_unlabeled = True
            
            self.optimizer.zero_grad()
            
            batch_loss = 0
            
            # 有标签数据的监督损失
            if has_labeled:
                labeled_outputs = self.model(labeled_imgs)
                supervised_loss = self.supervised_loss(labeled_outputs, labeled_masks)
                supervised_loss += dice_loss(labeled_outputs, labeled_masks)
                batch_loss += supervised_loss
                total_supervised_loss += supervised_loss.item()
            
            # 无标签数据的一致性损失
            if has_unlabeled:
                # 对学生模型应用更强的数据增强
                unlabeled_student = self.augment_unlabeled(unlabeled_imgs, strong=True)
                student_outputs = self.model(unlabeled_student)
                
                # 对教师模型应用标准数据增强
                with torch.no_grad():
                    # unlabeled_teacher = self.augment_unlabeled(unlabeled_imgs, strong=False)
                    unlabeled_teacher = unlabeled_imgs
                    teacher_outputs = self.ema_model(unlabeled_teacher)
                
                # 计算一致性损失
                consistency_loss = self.consistency_loss(
                    torch.sigmoid(student_outputs),
                    torch.sigmoid(teacher_outputs)
                )
                
                # 应用权重rampup
                current_weight = self.consistency_weight_rampup(epoch)
                consistency_loss *= current_weight
                
                batch_loss += consistency_loss
                total_consistency_loss += consistency_loss.item()
            
            batch_loss.backward()
            self.optimizer.step()
            
            # 更新教师模型
            self.update_ema(self.ema_decay)
            
            self.global_step += 1
            total_loss += batch_loss.item()
            
            pbar.set_postfix({
                'sup_loss': f'{total_supervised_loss/(i+1):.4f}',
                'cons_loss': f'{total_consistency_loss/(i+1):.4f}',
                'total_loss': f'{total_loss/(i+1):.4f}'
            })
        
        return total_supervised_loss / max_length, total_consistency_loss / max_length, total_loss / max_length
    
    def augment_unlabeled(self, images, strong=False):
        """对无标签数据进行增强"""
        batch_size = images.shape[0]
        augmented_images = []
        
        for i in range(batch_size):
            img = images[i]
            
            # 基础增强（对教师和学生都应用）
            if random.random() > 0.5:
                img = TF.hflip(img)
            if random.random() > 0.5:
                img = TF.vflip(img)
            
            # 强增强（只对学生应用）
            if strong:
                # 随机旋转
                angle = random.uniform(-10, 10)
                img = TF.rotate(img, angle)
                
                # 弹性变换（简化版）
                if random.random() > 0.5:
                    img = self.elastic_transform(img)
            
            augmented_images.append(img)
        
        return torch.stack(augmented_images)
    
    def elastic_transform(self, image, alpha=1, sigma=2):
        """简化的弹性变换"""
        # 这里实现一个简化的弹性变换
        # 在实际应用中可以使用更复杂的实现
        noise = torch.randn_like(image) * sigma
        return image
    
    def validate(self, val_loader):
        """验证模型性能"""
        self.model.eval()
        self.ema_model.eval()
        
        total_dice = 0
        total_samples = 0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                
                # 使用学生模型进行预测
                outputs = self.model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                dice = dice_coeff(preds.cpu(), masks.cpu())
                total_dice += dice * imgs.size(0)
                total_samples += imgs.size(0)
        
        return total_dice / total_samples


class LabeledDataset(Dataset):
    """有标签数据集（只使用原始domain1图像）"""
    def __init__(self):
        self.base = FazSegDataset(domain_id=1, split="train", aug="none")
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        return self.base[idx]


class UnlabeledDataset(Dataset):
    """无标签数据集（使用迁移图像）"""
    def __init__(self, use_feddg=True, use_cyclegan=True):
        self.base = Task3AugDataset(use_feddg=use_feddg, use_cyclegan=use_cyclegan, use_domain1=False)
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        # 只返回图像，不返回mask
        img, _ = self.base[idx]
        return img, torch.zeros_like(img)  # 返回假标签用于兼容


def train_mean_teacher():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # 数据集
    labeled_dataset = LabeledDataset()
    unlabeled_dataset = UnlabeledDataset(use_feddg=True, use_cyclegan=True)
    
    # 验证集（使用domain2的测试集）
    val_dataset = FazSegDataset(domain_id=4, split="test", aug="none")
    
    # 数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=4, shuffle=True, num_workers=2)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # 模型
    student_model = UNet2D(in_ch=1, num_classes=1).to(device)
    teacher_model = UNet2D(in_ch=1, num_classes=1).to(device)
    
    # 初始化教师模型与学生模型相同
    teacher_model.load_state_dict(student_model.state_dict())
    
    # 优化器
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
    
    # 训练器
    trainer = MeanTeacherTrainer(student_model, teacher_model, optimizer, device)
    
    # 训练参数
    epochs = 100
    best_dice = 0.0
    
    for epoch in range(epochs):
        # 训练
        sup_loss, cons_loss, total_loss = trainer.train_epoch(labeled_loader, unlabeled_loader, epoch)
        
        # 验证
        if epoch % 1 == 0:
            dice_score = trainer.validate(val_loader)
            print(f"Epoch {epoch}: Dice = {dice_score:.4f}")
            
            # 保存最佳模型
            if dice_score > best_dice:
                best_dice = dice_score
                torch.save(student_model.state_dict(), "checkpoints/task3_mean_teacher_student_best.pth")
                torch.save(teacher_model.state_dict(), "checkpoints/task3_mean_teacher_teacher_best.pth")
                print(f" >>> Saved new best model! Dice={dice_score:.4f}")
        
        # 保存检查点
        # if epoch % 10 == 0:
        #     torch.save(student_model.state_dict(), f"checkpoints/task3_mean_teacher_student_epoch{epoch}.pth")
        #     torch.save(teacher_model.state_dict(), f"checkpoints/task3_mean_teacher_teacher_epoch{epoch}.pth")
    
    print(f"Training completed. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    train_mean_teacher()