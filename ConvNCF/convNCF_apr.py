import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNCF_APR(nn.Module):
    def __init__(self, user_count, item_count, device, alpha=1.0, epsilon=0.5):
        super(ConvNCF_APR, self).__init__()
        
        self.device = device
        self.user_count = user_count
        self.item_count = item_count
        self.embedding_size = 64
        
        # APR 参数
        self.alpha = alpha     # 对抗 Loss 的权重
        self.epsilon = epsilon   # 扰动幅度
        
        # 1. Embedding 层
        self.P = nn.Embedding(self.user_count, self.embedding_size)
        self.Q = nn.Embedding(self.item_count, self.embedding_size)

        # 2. CNN 结构 (ConvNCF 部分)
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

        # 3. 全连接层 (ConvNCF 部分)
        self.fc = nn.Linear(32, 1)

    def _net_forward(self, user_emb, item_emb, is_pretrain):
        """
        内部核心计算逻辑：封装了 Pretrain 和 ConvNCF 的分支判断。
        这样在计算原始 Loss、计算梯度、计算对抗 Loss 时都可以复用这个逻辑。
        """
        # ==========================================
        # 这里的 if is_pretrain 就是您找的预训练处理逻辑
        # ==========================================
        if is_pretrain:
            # --- 预训练模式 (MF) ---
            # 直接计算 Embedding 的内积，不经过 CNN
            prediction = torch.sum(torch.mul(user_emb, item_emb), dim=1)
        else:
            # --- ConvNCF 模式 ---
            # 1. 计算外积 (Interaction Map)
            interaction_map = torch.bmm(user_emb.unsqueeze(2), item_emb.unsqueeze(1))
            interaction_map = interaction_map.view((-1, 1, self.embedding_size, self.embedding_size))
            
            # 2. 输入 CNN
            feature_map = self.cnn(interaction_map)
            feature_vec = feature_map.view((-1, 32))
            
            # 3. 输入 MLP
            prediction = self.fc(feature_vec)
            prediction = prediction.view((-1))
            
        return prediction

    def calculate_bpr_loss(self, pos_preds, neg_preds):
        """BPR Loss: -log(sigmoid(pos - neg))"""
        distance = pos_preds - neg_preds
        return torch.sum(F.softplus(-distance))

    def clear_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def get_perturbation(self, user_ids, pos_ids, neg_ids, is_pretrain):
        """计算对抗扰动"""
        user_emb = self.P(user_ids).detach()
        pos_emb = self.Q(pos_ids).detach()
        neg_emb = self.Q(neg_ids).detach()

        user_emb.requires_grad_(True)
        pos_emb.requires_grad_(True)
        neg_emb.requires_grad_(True)

        # 这里传入 is_pretrain，保证计算扰动时使用的网络结构与当前训练阶段一致
        pos_preds = self._net_forward(user_emb, pos_emb, is_pretrain)
        neg_preds = self._net_forward(user_emb, neg_emb, is_pretrain)

        loss = self.calculate_bpr_loss(pos_preds, neg_preds)
        loss.backward()

        def normalize(grad):
            return self.epsilon * grad / (torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)

        user_pert = normalize(user_emb.grad)
        pos_pert = normalize(pos_emb.grad)
        neg_pert = normalize(neg_emb.grad)

        self.clear_grad()
        return user_pert.detach(), pos_pert.detach(), neg_pert.detach()

    def forward(self, user_ids, pos_ids, neg_ids, is_pretrain=False, user_adv=False):
        """
        前向传播
        :param is_pretrain: 关键参数，决定是走 MF 逻辑还是 CNN 逻辑
        """
        # 1. 获取 Embedding
        user_emb = self.P(user_ids)
        pos_emb = self.Q(pos_ids)
        neg_emb = self.Q(neg_ids)

        # 2. 正常计算 Loss (Original)
        # 调用 _net_forward，它会根据 is_pretrain 自动选择路径
        pos_preds = self._net_forward(user_emb, pos_emb, is_pretrain)
        neg_preds = self._net_forward(user_emb, neg_emb, is_pretrain)
        
        ori_loss = self.calculate_bpr_loss(pos_preds, neg_preds)

        total_loss = ori_loss

        # 3. 对抗计算 (APR)
        if user_adv:
            # 获取扰动 (同样传入 is_pretrain)
            user_pert, pos_pert, neg_pert = self.get_perturbation(user_ids, pos_ids, neg_ids, is_pretrain)
            
            # 加噪
            p_user_emb = user_emb + user_pert
            p_pos_emb = pos_emb + pos_pert
            p_neg_emb = neg_emb + neg_pert
            
            # 对抗计算
            adv_pos_preds = self._net_forward(p_user_emb, p_pos_emb, is_pretrain)
            adv_neg_preds = self._net_forward(p_user_emb, p_neg_emb, is_pretrain)
            
            adv_loss = self.calculate_bpr_loss(adv_pos_preds, adv_neg_preds)
            total_loss = ori_loss + self.alpha * adv_loss

        return total_loss, pos_preds, neg_preds
    
    def predict(self, user_ids, item_ids, is_pretrain=False):
        """
        用于推理/测试阶段，计算用户对物品的评分。
        
        Args:
            user_ids: tensor (Batch,) 用户ID
            item_ids: tensor (Batch,) 物品ID
            is_pretrain: bool, 关键参数
                         True -> 使用 MF (内积) 逻辑 (对应 epoch < 99)
                         False -> 使用 ConvNCF (CNN) 逻辑 (对应 epoch >= 99)
        Returns:
            prediction: tensor (Batch,) 预测分数 (Logits)
        """
        # 1. 确保输入是 Tensor 并且在正确的设备上
        if not torch.is_tensor(user_ids):
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        if not torch.is_tensor(item_ids):
            item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)
            
        # 2. 获取 Embedding
        user_emb = self.P(user_ids)
        item_emb = self.Q(item_ids)

        # 3. 复用 _net_forward 进行计算
        # 这个函数会根据 is_pretrain 自动选择走 MF 还是 CNN
        return self._net_forward(user_emb, item_emb, is_pretrain)