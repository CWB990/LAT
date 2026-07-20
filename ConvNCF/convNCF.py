import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count, device):
        super(ConvNCF, self).__init__()

        # some variables
        self.device = device
        self.user_count = user_count
        self.item_count = item_count
        # self.item_count = 12929
        # embedding setting
        self.embedding_size = 64

        self.P = nn.Embedding(self.user_count, self.embedding_size).to(self.device)
        self.Q = nn.Embedding(self.item_count, self.embedding_size).to(self.device)

        # cnn setting
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        self.cnn = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        ).to(self.device)

        self.fc = nn.Linear(32, 1).to(self.device)
        
    def _calc_bpr_loss(self, pos_preds, neg_preds):
        """BPR Loss: -log(sigmoid(pos - neg))"""
        distance = pos_preds - neg_preds
        return F.softplus(-distance)
    
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
    
    def forward(self, user_ids, pos_ids, neg_ids, is_pretrain=False, reduction='sum'):
        # 兼容性处理
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids).long().to(self.device)
        if not isinstance(pos_ids, torch.Tensor):
            pos_ids = torch.tensor(pos_ids).long().to(self.device)
        if not isinstance(neg_ids, torch.Tensor):
            neg_ids = torch.tensor(neg_ids).long().to(self.device)

        # 1. 获取原始 Embedding
        user_emb = self.P(user_ids)
        pos_emb = self.Q(pos_ids)
        neg_emb = self.Q(neg_ids)

        # 2. 计算原始 Loss
        # 自动根据 is_pretrain 选择路径
        pos_preds = self._net_forward(user_emb, pos_emb, is_pretrain)
        neg_preds = self._net_forward(user_emb, neg_emb, is_pretrain)
        loss_vector = self._calc_bpr_loss(pos_preds, neg_preds)
        
        # 计算 Loss (reduce=mean for optimization)
         # --- 根据 reduction 参数决定返回什么 ---
        if reduction == 'sum':
            return torch.sum(loss_vector)
        elif reduction == 'mean':
            return torch.mean(loss_vector)
        elif reduction == 'none':
            return loss_vector  # AWP 需要这个！
        else:
            # 默认行为（防止传错）
            return torch.sum(loss_vector)

    
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
