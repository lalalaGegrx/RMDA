import sys
sys.path.append('/home/zju/Python_Scripts/EEGemotion')

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eigvalsh
from scipy.linalg import sqrtm, logm, expm

from DRBN.Model_SPDNet.spd import SPDTangentSpaceFunction, SPDUnTangentSpaceFunction


# ============= 黎曼几何基础操作 =============
class RiemannianOperations:
    @staticmethod
    def sym(X):
        """确保矩阵对称"""
        return 0.5 * (X + X.transpose(-1, -2))
    
    @staticmethod
    def logm(X):
        """矩阵对数映射"""
        s, U = torch.linalg.eigh(X)
        return U @ torch.diag_embed(torch.log(s.clamp(min=1e-10))) @ U.transpose(-1, -2)
    
    @staticmethod
    def expm(X):
        """矩阵指数映射"""
        s, U = torch.linalg.eigh(X)
        return U @ torch.diag_embed(torch.exp(s)) @ U.transpose(-1, -2)
    
    @staticmethod
    def riemannian_mean(X):
        barycenter = torch.mean(X, dim=0, keepdim=True)  # 初始化为算术平均
        try:
            eigvals, eigvcts = torch.linalg.eigh(barycenter[0])
            eigvals_inv, eigvcts_inv = torch.linalg.eigh(torch.linalg.inv(barycenter[0]))
            barycenter_sqrtm = eigvcts @ torch.diag(torch.sqrt(eigvals)) @ eigvcts.t()
            barycenter_invsqrtm = eigvcts_inv @ torch.diag(torch.sqrt(eigvals_inv)) @ eigvcts_inv.t()
        except:
            return barycenter
        
        for _ in range(100):
            try:
                tangent_vectors = []
                for x in X:
                    eigvals, eigvcts = torch.linalg.eigh(barycenter_invsqrtm @ x @ barycenter_invsqrtm)
                    barycenter_logm = eigvcts @ torch.diag(torch.log(eigvals.clamp(min=1e-10))) @ eigvcts.t()
                    tangent_vectors.append(barycenter_sqrtm @ barycenter_logm @ barycenter_sqrtm)
                tangent_vectors = torch.stack(tangent_vectors)

                # 计算切空间中的平均值
                mean_tangent = torch.mean(tangent_vectors, dim=0)
                
                # 映射回流形 (公式5)
                eigvals, eigvcts = torch.linalg.eigh(barycenter_invsqrtm @ mean_tangent @ barycenter_invsqrtm)
                barycenter_expm = eigvcts @ torch.diag(torch.exp(eigvals.clamp(min=1e-10))) @ eigvcts.t()
                new_bary = barycenter_sqrtm @ barycenter_expm @ barycenter_sqrtm

                # 检查收敛
                error = torch.norm(new_bary - barycenter)
                if error < 1e-4:
                    break
                barycenter = new_bary.float()
            except:
                barycenter = torch.mean(X, dim=0, keepdim=True)
                break
        
        return barycenter
    
    @staticmethod
    def riemannian_distance(X1, X2):
        """黎曼距离 (AIRM)"""
        inv_sqrt_X1 = torch.linalg.inv(torch.linalg.cholesky(X1+(torch.eye(X1.shape[1])*1e-3).to(X1.device))).transpose(-1, -2)
        M = inv_sqrt_X1 @ X2 @ inv_sqrt_X1
        eigenvalues = torch.linalg.eigvalsh(M)
        return torch.norm(torch.log(eigenvalues.clamp(min=1e-10)), dim=-1)
    
    @staticmethod
    def parallel_transport(S, X1, X2):
        """平行传输 (公式10)"""
        inv_X1 = torch.linalg.inv(X1)
        sqrt_inv_X1 = torch.linalg.cholesky(inv_X1).transpose(-1, -2)
        term = torch.linalg.cholesky(X2 @ inv_X1).transpose(-1, -2)
        return term @ sqrt_inv_X1 @ S @ sqrt_inv_X1.transpose(-1, -2) @ term.transpose(-1, -2)

# ============= 重心归一化层 (Barycenter Normalization) =============
class BarycenterNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, device='cpu'):
        """
        num_features: SPD矩阵的特征数 (c)
        eps: 数值稳定性常数
        momentum: 运行平均的动量
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer('running_mean', torch.eye(num_features)[None, ...])
        
    def forward(self, X):
        """
        X: 输入SPD矩阵 (batch_size, c, c)
        返回: 归一化的SPD矩阵
        """
        if self.training:
            # 计算当前batch的黎曼重心 (公式6)
            batch_mean = self.compute_barycenter(X)
            
            # 更新运行平均值
            self.running_mean = self.running_mean * (1 - self.momentum) + batch_mean.detach() * self.momentum
        else:
            batch_mean = self.running_mean
        
        # 归一化过程 (公式11)
        normalized = self.normalize(X, batch_mean)
        return normalized
    
    def compute_barycenter(self, X, max_iter=50, tol=1e-5):
        """迭代计算黎曼重心 (Karcher Flow算法)"""
        # batch_size, c, _ = X.shape
        
        # return barycenter
        pass
    
    def normalize(self, X, barycenter):
        """归一化SPD矩阵 (公式11)"""
        normalized_list = []
        for i in range(X.shape[0]):
            # 1. 投影到切空间
            # S_i = RiemannianOperations.logm(barycenter[0] @ X[i])
            eigvals, eigvcts = torch.linalg.eigh(barycenter[0])
            eigvals_inv, eigvcts_inv = torch.linalg.eigh(torch.linalg.inv(barycenter[0]))
            barycenter_sqrtm = eigvcts @ torch.diag(torch.sqrt(eigvals)) @ eigvcts.t()
            barycenter_invsqrtm = eigvcts_inv @ torch.diag(torch.sqrt(eigvals_inv)) @ eigvcts_inv.t()
            eigvals, eigvcts = torch.linalg.eigh(barycenter_invsqrtm @ X[i] @ barycenter_invsqrtm)
            barycenter_logm = eigvcts @ torch.diag(torch.log(eigvals.clamp(min=1e-10))) @ eigvcts.t()
            S_i = barycenter_sqrtm @ barycenter_logm @ barycenter_sqrtm
            
            # 2. 平行传输 (公式10)
            S_i_transported = RiemannianOperations.parallel_transport(
                S_i, barycenter[0], torch.eye(barycenter.shape[-1]))
            
            # 3. 映射回流形
            X_i_normalized = RiemannianOperations.expm(S_i_transported)
            normalized_list.append(X_i_normalized)
        
        return torch.stack(normalized_list)



# ============= 重心分类层 (Barycenter Classification) =============
class BarycenterClassification(nn.Module):
    def __init__(self, num_classes, lambda1=0.001, lambda2=0.5, alpha=0.1):
        """
        num_classes: 类别数量
        lambda1, lambda2: 损失权重参数
        alpha: 重心学习率
        """
        super().__init__()
        self.num_classes = num_classes
        self.lambda1 = lambda1
    
        # 运行统计量
        self.register_buffer('running_mean_dist', torch.zeros(num_classes))
        self.register_buffer('running_std_dist', torch.ones(num_classes))
    
    def forward(self, X, out, labels):
        """
        X: 归一化的SPD矩阵 (batch_size, c, c)
        labels: 样本标签 (batch_size,)
        返回: 总损失值
        """
        batch_size = X.shape[0]
        
        
        unique_labels = torch.unique(labels)
        class_barycenters = torch.zeros((len(unique_labels), X.shape[1], X.shape[1]))
        for label in unique_labels:
            mask = (labels == label)
            class_samples = X[mask]
            class_barycenters[label.long()] = RiemannianOperations.riemannian_mean(class_samples)
        class_barycenters = class_barycenters.to(X.device)
        
        # 计算类内损失 (公式13)
        intra_loss = self.compute_intra_loss(class_barycenters, X, labels)
        
        # 计算类间损失 (公式14)
        inter_loss = self.compute_inter_loss(class_barycenters, X, out, labels)
        
        # 组合损失 (公式12)
        total_loss = intra_loss + inter_loss
        return total_loss
    
    def compute_intra_loss(self, class_barycenters, X, labels):
        """计算类内损失 (公式13)"""
        losses = []
        for i in range(X.shape[0]):
            bary = class_barycenters[int(labels[i])]
            dist = RiemannianOperations.riemannian_distance(X[i], bary)
            losses.append(dist)
        return self.lambda1 * torch.mean(torch.stack(losses))
    
    def compute_inter_loss(self, class_barycenters, X, out, labels):
        """计算类间损失 (公式14)"""
        # 第一部分: 基于分散度的损失
        disp_loss = 0.0
        for i in range(X.shape[0]):
            for j in range(self.num_classes):
                if j != labels[i]:
                    d_ij = RiemannianOperations.riemannian_distance(X[i], class_barycenters[j])
                    disp_loss += d_ij / (X.shape[0] * (self.num_classes-1))
        
        # 第二部分: 标准交叉熵损失
        # logits = self.compute_logits(X)  # 将SPD矩阵转换为logits
        ce_loss = nn.CrossEntropyLoss()(out, labels.long())
        
        return -self.lambda1 * disp_loss + ce_loss

    
    def compute_logits(self, X):
        """将SPD矩阵转换为分类logits"""
        # 简化实现: 使用矩阵的弗罗贝尼乌斯范数作为特征
        batch_size, c, _ = X.shape
        features = torch.norm(X, dim=(1, 2), p='fro')  # (batch_size,)
        return features.unsqueeze(1).repeat(1, self.num_classes)  # 伪logits


# ============= 黎曼重心网络块 =============
class RiemannianBarycenterBlock(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.norm_layer = BarycenterNorm(num_features)
        self.class_layer = BarycenterClassification(num_classes)
    
    def forward(self, X, out, labels):
        """
        X: 输入SPD矩阵 (batch_size, c, c)
        labels: 样本标签 (batch_size,)
        返回: 
          normalized: 归一化后的SPD矩阵
          total_loss: 组合损失
        """
        # normalized = self.norm_layer(X)
        total_loss = self.class_layer(X, out, labels)

        #total_loss = nn.CrossEntropyLoss()(out, labels.long())

        return total_loss

