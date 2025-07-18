import numpy as np
from scipy.linalg import logm, expm, eigvals
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

# 黎曼空间工具函数
def riemannian_distance(P1, P2):
    """计算两个SPD矩阵的黎曼距离"""
    eig_vals = eigvals(np.linalg.inv(P1) @ P2)
    return np.sqrt(np.sum(np.log(eig_vals)**2))

def riemannian_mean(covariances, max_iter=50, tol=1e-6):
    """计算SPD矩阵集的黎曼均值"""
    N, d, _ = covariances.shape
    G = np.mean(covariances, axis=0)  # 初始化为欧氏均值
    
    for _ in range(max_iter):
        # 计算切空间向量总和
        tangents = np.zeros_like(G)
        for C in covariances:
            invG = np.linalg.inv(G)
            tangents += logm(invG @ C @ invG)
        tangents /= N
        
        # 指数映射更新均值
        new_G = G @ expm(tangents) @ G
        
        # 检查收敛
        if np.linalg.norm(new_G - G) < tol:
            break
        G = new_G
    return G

# 黎曼分类器实现
class MDRM(BaseEstimator, ClassifierMixin):
    """最小黎曼距离均值分类器"""
    def __init__(self):
        self.class_centers_ = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_centers_ = []
        for c in self.classes_:
            class_data = X[y == c]
            self.class_centers_.append(riemannian_mean(class_data))
        return self

    def predict(self, X):
        preds = []
        for P in X:
            dists = [riemannian_distance(P, C) for C in self.class_centers_]
            preds.append(self.classes_[np.argmin(dists)])
        return np.array(preds)

class GLRSQ(BaseEstimator, ClassifierMixin):
    """广义黎曼空间量化分类器"""
    def __init__(self, n_prototypes=1, lr=0.1, max_epochs=10):
        self.n_prototypes = n_prototypes
        self.lr = lr
        self.max_epochs = max_epochs
        self.prototypes = None
        self.prototype_labels = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.prototypes = []
        self.prototype_labels = []
        
        # 初始化原型：每类选择n_prototypes个样本
        for c in self.classes_:
            class_data = X[y == c]
            idx = np.random.choice(len(class_data), self.n_prototypes, replace=False)
            self.prototypes.extend(class_data[idx])
            self.prototype_labels.extend([c] * self.n_prototypes)
        
        # 优化原型
        for epoch in range(self.max_epochs):
            for i, P in enumerate(X):
                # 找到最近正确/错误原型
                dists = [riemannian_distance(P, W) for W in self.prototypes]
                sorted_idx = np.argsort(dists)
                
                # 寻找最近正确原型
                j = None
                for idx in sorted_idx:
                    if self.prototype_labels[idx] == y[i]:
                        j = idx
                        break
                
                # 寻找最近错误原型
                k = None
                for idx in sorted_idx:
                    if self.prototype_labels[idx] != y[i]:
                        k = idx
                        break
                
                if j is None or k is None:
                    continue
                
                # 计算梯度
                delta_j = dists[j]
                delta_k = dists[k]
                scale = 4 * delta_k / (delta_j + delta_k)**2
                grad_j = -scale * logm(np.linalg.inv(self.prototypes[j]) @ P)
                
                scale = 4 * delta_j / (delta_j + delta_k)**2
                grad_k = scale * logm(np.linalg.inv(self.prototypes[k]) @ P)
                
                # 更新原型
                self.prototypes[j] = self.prototypes[j] @ expm(self.lr * grad_j)
                self.prototypes[k] = self.prototypes[k] @ expm(self.lr * grad_k)
        return self

    def predict(self, X):
        preds = []
        for P in X:
            dists = [riemannian_distance(P, W) for W in self.prototypes]
            idx = np.argmin(dists)
            preds.append(self.prototype_labels[idx])
        return np.array(preds)

# PSO维度选择核心实现
class DSSR:
    def __init__(self, 
                 classifier='mdr', 
                 n_particles=20, 
                 max_iter=30, 
                 h_threshold=0.6, 
                 alpha=0.001,
                 omega=0.73, 
                 c1=1.5, 
                 c2=1.5, 
                 v_max=0.1):
        self.classifier_type = classifier
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.h = h_threshold
        self.alpha = alpha
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.best_position = None
        self.best_fitness = -np.inf
        self.channel_importance_ = None

    def _evaluate_fitness(self, particle, X_train, y_train, X_val, y_val):
        """评估粒子的适应度"""
        # 确定选择的通道 (x_i >= h)
        selected = particle >= self.h
        n_selected = np.sum(selected)
        
        if n_selected < 2:  # 至少需要2个通道
            return -np.inf
        
        # 降维：保留选中的行/列
        X_train_red = X_train[:, selected][:, :, selected]
        X_val_red = X_val[:, selected][:, :, selected]
        
        # 训练分类器
        if self.classifier_type == 'mdr':
            clf = MDRM()
        else:  # 'glrq'
            clf = GLRSQ(n_prototypes=1, max_epochs=5)
        
        clf.fit(X_train_red, y_train)
        preds = clf.predict(X_val_red)
        
        # 计算kappa值
        kappa = cohen_kappa_score(y_val, preds)
        if np.isnan(kappa):
            kappa = 0
        
        # 适应度函数 (Eq.12)
        reduction_ratio = 1 - n_selected / particle.shape[0]
        fitness = (1 - self.alpha) * reduction_ratio + self.alpha * kappa
        return fitness

    def fit(self, X_train, y_train, X_val, y_val):
        """PSO优化维度选择"""
        n_channels = X_train.shape[1]
        particles = np.random.uniform(0, 1, (self.n_particles, n_channels))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.n_particles, n_channels))
        pbest_positions = particles.copy()
        pbest_fitnesses = np.full(self.n_particles, -np.inf)
        
        for epoch in range(self.max_iter):
            for i in range(self.n_particles):
                # 评估当前粒子
                fitness = self._evaluate_fitness(
                    particles[i], X_train, y_train, X_val, y_val
                )
                
                # 更新个体最优
                if fitness > pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fitness
                    pbest_positions[i] = particles[i].copy()
                
                # 更新全局最优
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = particles[i].copy()
            
            # 更新粒子速度和位置
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.omega * velocities[i] +
                                 self.c1 * r1 * (pbest_positions[i] - particles[i]) +
                                 self.c2 * r2 * (self.best_position - particles[i]))
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 1)
        
        # 保存通道重要性
        self.channel_importance_ = self.best_position
        return self

    def transform(self, X):
        """应用维度选择"""
        selected = self.channel_importance_ >= self.h
        return X[:, selected][:, :, selected]
