import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNCF_VAT(nn.Module):
    def __init__(self, user_count, item_count, device, alpha=1.0, epsilon=0.5, user_lmb=2.0):
        super(ConvNCF_VAT, self).__init__()
        
        # --- 基础配置 ---
        self.device = device
        self.user_count = user_count
        self.item_count = item_count
        self.embedding_size = 64
        
        # --- VAT 特有参数 ---
        self.alpha = alpha       # 对抗 Loss 权重
        self.epsilon = epsilon   # 扰动幅度
        self.user_lmb = user_lmb # 用户自适应权重的 lambda
        
        # 1. Embedding 层
        self.P = nn.Embedding(self.user_count, self.embedding_size)
        self.Q = nn.Embedding(self.item_count, self.embedding_size)

        # 2. CNN 结构
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides), nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides), nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides), nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides), nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides), nn.ReLU(),
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides), nn.ReLU(),
        )

        # 3. 全连接层
        self.fc = nn.Linear(32, 1)

    def _net_forward(self, user_emb, item_emb, is_pretrain):
        """
        核心网络前向传播：
        根据 is_pretrain 决定走 MF (内积) 还是 ConvNCF (外积+CNN)
        """
        if is_pretrain:
            # --- 预训练模式 (MF) ---
            # 简单内积，不经过 CNN
            prediction = torch.sum(torch.mul(user_emb, item_emb), dim=1)
        else:
            # --- ConvNCF 模式 ---
            # 1. 外积 (Interaction Map)
            interaction_map = torch.bmm(user_emb.unsqueeze(2), item_emb.unsqueeze(1))
            interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))
            
            # 2. CNN
            feature_map = self.cnn(interaction_map)
            feature_vec = feature_map.view((-1, 32))
            
            # 3. MLP
            prediction = self.fc(feature_vec)
            prediction = prediction.view((-1))
            
        return prediction

    def _calc_bpr_loss(self, pos_preds, neg_preds):
        """BPR Loss: Softplus(-distance)"""
        distance = pos_preds - neg_preds
        return F.softplus(-distance)

    def _get_user_eps(self, user_list, loss, lmabda):
        """
        VAT 核心：计算用户自适应 epsilon 权重
        """
        loss = loss.detach()
        # 确保 user_list 是 Tensor
        if not torch.is_tensor(user_list):
            user_list = torch.tensor(user_list, device=loss.device)
        
        unique_users, inverse_indices, counts = torch.unique(
            user_list, return_inverse=True, return_counts=True)
        
        user_loss_sum = torch.zeros_like(unique_users, dtype=loss.dtype)
        user_loss_sum.index_add_(0, inverse_indices, loss)
        
        mean_loss = user_loss_sum.mean()
        user_total_loss = user_loss_sum[inverse_indices]
        
        # 避免除 0
        eps = mean_loss / (user_total_loss + 1e-9) - 1
        return lmabda * torch.sigmoid(eps)

    def clear_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def get_perturbation(self, user_ids, pos_ids, neg_ids, is_pretrain):
        """
        计算 VAT 扰动
        """
        # 1. 复制 Embedding 并开启梯度
        user_emb = self.P(user_ids).detach()
        pos_emb = self.Q(pos_ids).detach()
        neg_emb = self.Q(neg_ids).detach()

        user_emb.requires_grad_(True)
        pos_emb.requires_grad_(True)
        neg_emb.requires_grad_(True)

        # 2. 前向传播计算原始 Loss (复用 _net_forward)
        pos_preds = self._net_forward(user_emb, pos_emb, is_pretrain)
        neg_preds = self._net_forward(user_emb, neg_emb, is_pretrain)

        # 注意：这里计算的是 per-sample loss，不取 sum 或 mean，以便计算 user_eps
        loss_vec = self._calc_bpr_loss(pos_preds, neg_preds)
        
        # 3. 反向传播
        loss_vec.mean().backward()

        # 4. 计算用户自适应权重 (VAT 特有)
        user_eps = self._get_user_eps(user_ids, loss_vec, self.user_lmb)

        # 5. 生成归一化扰动
        def normalize(grad):
            return self.epsilon * grad / (torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)

        user_pert = normalize(user_emb.grad)
        pos_pert = normalize(pos_emb.grad)
        neg_pert = normalize(neg_emb.grad)

        # 6. 应用自适应权重
        user_eps_view = user_eps.view(-1, 1)
        user_pert = user_pert * user_eps_view
        pos_pert = pos_pert * user_eps_view
        neg_pert = neg_pert * user_eps_view

        # 清理梯度，防止影响主优化器
        self.clear_grad()
        
        return user_pert.detach(), pos_pert.detach(), neg_pert.detach()

    def forward(self, user_ids, pos_ids, neg_ids, is_pretrain=False, user_adv=False):
        """
        Args:
            user_ids, pos_ids, neg_ids: 输入索引
            is_pretrain (bool): True=使用MF, False=使用CNN
            user_adv (bool): True=开启VAT训练
        """
        # 1. 获取原始 Embedding
        user_emb = self.P(user_ids)
        pos_emb = self.Q(pos_ids)
        neg_emb = self.Q(neg_ids)

        # 2. 计算原始 Loss
        # 自动根据 is_pretrain 选择路径
        pos_preds = self._net_forward(user_emb, pos_emb, is_pretrain)
        neg_preds = self._net_forward(user_emb, neg_emb, is_pretrain)
        
        # 计算 Loss (reduce=mean for optimization)
        ori_loss_vec = self._calc_bpr_loss(pos_preds, neg_preds)
        ori_loss = ori_loss_vec.sum() # BPR通常sum或者mean都可以，根据习惯调整

        total_loss = ori_loss

        # 3. VAT 对抗训练
        if user_adv:
            # 获取扰动 (传入 is_pretrain 确保计算梯度时网络结构一致)
            u_noise, p_noise, n_noise = self.get_perturbation(user_ids, pos_ids, neg_ids, is_pretrain)
            
            # 加入扰动
            p_user_emb = user_emb + u_noise
            p_pos_emb = pos_emb + p_noise
            p_neg_emb = neg_emb + n_noise
            
            # 计算对抗 Loss (复用 _net_forward)
            adv_pos_preds = self._net_forward(p_user_emb, p_pos_emb, is_pretrain)
            adv_neg_preds = self._net_forward(p_user_emb, p_neg_emb, is_pretrain)
            
            adv_loss_vec = self._calc_bpr_loss(adv_pos_preds, adv_neg_preds)
            adv_loss = adv_loss_vec.sum()
            
            total_loss = ori_loss + self.alpha * adv_loss

        # 返回 Loss 和 预测值 (兼容你的评估逻辑)
        return total_loss, pos_preds, neg_preds
    
    def predict(self, user_ids, item_ids, is_pretrain=False):
        """
        推理/测试接口
        is_pretrain: 决定推理时使用 MF 还是 CNN
        """
        if not torch.is_tensor(user_ids):
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        if not torch.is_tensor(item_ids):
            item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)
            
        user_emb = self.P(user_ids)
        item_emb = self.Q(item_ids)

        return self._net_forward(user_emb, item_emb, is_pretrain)