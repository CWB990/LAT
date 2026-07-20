import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN_APR(nn.Module):
    # [修改] 增加 alpha 和 epsilon 参数
    def __init__(self, user_count, item_count, device, adj_mat, latent_dim=64, n_layers=3, keep_prob=1.0, alpha=1.0, epsilon=0.5):
        super(LightGCN_APR, self).__init__()

        self.user_count = user_count
        self.item_count = item_count
        self.device = device
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.keep_prob = keep_prob
        
        # [新增] APR 参数
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.Graph = adj_mat.to(device)

        self.embedding_user = nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def _dropout_graph(self, graph, keep_prob):
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        return torch.sparse.FloatTensor(index.t(), values, size).to(self.device)

    # [修改] 增加参数，允许传入外部的 embedding (比如加了扰动的 embedding)
    # 如果 user_emb_inp 为 None，则默认使用模型自带的 embedding
    def _propagate(self, user_emb_inp=None, item_emb_inp=None):
        """核心图卷积过程"""
        if user_emb_inp is None:
            users_emb = self.embedding_user.weight
        else:
            users_emb = user_emb_inp
        
        if item_emb_inp is None:
            items_emb = self.embedding_item.weight
        else:
            items_emb = item_emb_inp
            
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        if self.training and self.keep_prob < 1.0:
            g_droped = self._dropout_graph(self.Graph, self.keep_prob)
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        light_out = torch.mean(torch.stack(embs, dim=1), dim=1)
        
        users, items = torch.split(light_out, [self.user_count, self.item_count])
        return users, items

    def _calc_bpr_loss(self, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0):
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        
        # BPR Loss
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # Reg Loss (仅基于初始 Embedding 计算正则，对抗样本通常不计入正则)
        reg_loss = (1/2) * (userEmb0.norm(2).pow(2) + 
                             posEmb0.norm(2).pow(2) + 
                             negEmb0.norm(2).pow(2)) / float(len(users_emb))
        
        return loss, reg_loss

    # [新增] 辅助函数：清空梯度
    def clear_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # [新增] 计算对抗扰动 (核心 APR 逻辑)
    def get_perturbation(self, user_ids, pos_ids, neg_ids):
        """
        计算使得 Loss 最大的扰动方向
        注意：LightGCN 是全图传播，为了计算方便，我们对整个 Embedding 矩阵计算扰动，
        但在 loss backward 时，只有 batch 涉及的节点会有梯度。
        """
        # 1. 获取当前的 Embedding 副本，并开启梯度追踪
        user_emb_0 = self.embedding_user.weight.detach()
        item_emb_0 = self.embedding_item.weight.detach()
        
        user_emb_0.requires_grad_(True)
        item_emb_0.requires_grad_(True)
        
        # 2. 使用这些临时 Embedding 进行图传播
        # 注意：这里调用 _propagate 时传入了临时变量
        all_users, all_items = self._propagate(user_emb_0, item_emb_0)
        
        u_g_embeddings = all_users[user_ids]
        pos_i_g_embeddings = all_items[pos_ids]
        neg_i_g_embeddings = all_items[neg_ids]

        # 3. 计算 BPR Loss (不需要正则项来计算对抗梯度，只看 ranking 误差)
        pos_scores = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # 4. 反向传播，获取梯度
        loss.backward()
        
        # 5. 计算扰动 (L2 Normalize * epsilon)
        # 只有在 batch 中出现过的节点（及其多跳邻居）才会有梯度，其他为 0
        def normalize_and_scale(grad):
            if grad is None: return 0
            # 加上 1e-8 防止除以 0
            norm = torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8
            return self.epsilon * (grad / norm)

        user_pert = normalize_and_scale(user_emb_0.grad)
        item_pert = normalize_and_scale(item_emb_0.grad)
        
        # 清理梯度，防止影响正常的 optimizer.step()
        self.clear_grad()
        
        # 返回整个 Embedding 矩阵的扰动 (大部分行是 0)
        return user_pert.detach(), item_pert.detach()

    # [修改] Forward 增加 user_adv 参数
    def forward(self, user_ids, pos_ids, neg_ids, decay=1e-4, user_adv=False):
        # 确保 ID 为 Tensor
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids).long().to(self.device)
        if not isinstance(pos_ids, torch.Tensor):
            pos_ids = torch.tensor(pos_ids).long().to(self.device)
        if not isinstance(neg_ids, torch.Tensor):
            neg_ids = torch.tensor(neg_ids).long().to(self.device)

        # ====================
        # 1. 正常前向传播 (Original Loss)
        # ====================
        all_users, all_items = self._propagate() # 使用默认权重
        
        u_g_embeddings = all_users[user_ids]
        pos_i_g_embeddings = all_items[pos_ids]
        neg_i_g_embeddings = all_items[neg_ids]

        u_ego_embeddings = self.embedding_user(user_ids)
        pos_ego_embeddings = self.embedding_item(pos_ids)
        neg_ego_embeddings = self.embedding_item(neg_ids)

        bpr_loss, reg_loss = self._calc_bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,
            u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings
        )
        
        ori_loss = bpr_loss + decay * reg_loss
        total_loss = ori_loss

        # ====================
        # 2. 对抗训练 (APR)
        # ====================
        if user_adv:
            # A. 计算扰动
            user_pert, item_pert = self.get_perturbation(user_ids, pos_ids, neg_ids)
            
            # B. 将扰动加到原始 Embedding 上
            # 注意：不直接修改 self.embedding.weight，而是创建新的 Tensor 传给 propagate
            p_user_emb = self.embedding_user.weight + user_pert
            p_item_emb = self.embedding_item.weight + item_pert
            
            # C. 使用加噪后的 Embedding 进行图传播
            p_all_users, p_all_items = self._propagate(p_user_emb, p_item_emb)
            
            # D. 取出 Batch 对应的加噪 Embedding
            p_u_g = p_all_users[user_ids]
            p_pos_g = p_all_items[pos_ids]
            p_neg_g = p_all_items[neg_ids]
            
            # E. 计算对抗 Loss (通常对抗阶段只算 Ranking Loss，不算正则)
            p_pos_scores = torch.sum(torch.mul(p_u_g, p_pos_g), dim=1)
            p_neg_scores = torch.sum(torch.mul(p_u_g, p_neg_g), dim=1)
            adv_loss = torch.mean(F.softplus(p_neg_scores - p_pos_scores))
            
            # F. 总 Loss 结合
            total_loss = ori_loss + self.alpha * adv_loss

        return total_loss

    def predict(self, user_ids, item_ids):
        """保持不变"""
        if not torch.is_tensor(user_ids):
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        if not torch.is_tensor(item_ids):
            item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)

        all_users, all_items = self._propagate()
        
        u_emb = all_users[user_ids]
        i_emb = all_items[item_ids]
        
        prediction = torch.sum(torch.mul(u_emb, i_emb), dim=1)
        return prediction

    def get_all_ratings(self, user_ids):
        """保持不变"""
        all_users, all_items = self._propagate()
        u_emb = all_users[user_ids.long()]
        ratings = torch.matmul(u_emb, all_items.t())
        return torch.sigmoid(ratings)