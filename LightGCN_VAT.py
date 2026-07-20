import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN_VAT(nn.Module):
    def __init__(self, user_count, item_count, device, adj_mat, 
                 latent_dim=64, n_layers=3, keep_prob=1.0,
                 alpha=1.0, epsilon=0.5, user_lmb=2.0):
        super(LightGCN_VAT, self).__init__()

        # --- 基础配置 ---
        self.user_count = user_count
        self.item_count = item_count
        self.device = device
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.keep_prob = keep_prob
        
        # --- VAT 特有参数 ---
        self.alpha = alpha       # 对抗 Loss 权重
        self.epsilon = epsilon   # 扰动幅度
        self.user_lmb = user_lmb # 用户自适应权重的 lambda

        # 图结构
        self.Graph = adj_mat.to(device)

        # Embedding 层 (Ego Embeddings)
        self.embedding_user = nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.latent_dim)

        # 初始化
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def _dropout_graph(self, graph, keep_prob):
        """对稀疏图进行 Dropout"""
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        return torch.sparse.FloatTensor(index.t(), values, size).to(self.device)

    def _propagate(self):
        """图卷积传播，返回最终的所有 User 和 Item Embeddings"""
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        if self.training and self.keep_prob < 1.0:
            g_droped = self._dropout_graph(self.Graph, self.keep_prob)
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.user_count, self.item_count])
        return users, items

    def _calc_pure_bpr_loss(self, users_emb, pos_emb, neg_emb, reduction='mean'):
        """
        仅计算 BPR Loss，不包含正则项。
        用于 VAT 扰动梯度的计算和对抗 Loss。
        """
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        
        loss_vec = F.softplus(neg_scores - pos_scores)
        
        if reduction == 'mean':
            return loss_vec.mean()
        elif reduction == 'sum':
            return loss_vec.sum()
        else:
            return loss_vec # return vector for user_eps calculation

    def _get_user_eps(self, user_list, loss, lmabda):
        """
        VAT 核心：计算用户自适应 epsilon 权重
        """
        loss = loss.detach()
        if not torch.is_tensor(user_list):
            user_list = torch.tensor(user_list, device=loss.device)
        
        unique_users, inverse_indices, counts = torch.unique(
            user_list, return_inverse=True, return_counts=True)
        
        user_loss_sum = torch.zeros_like(unique_users, dtype=loss.dtype)
        user_loss_sum.index_add_(0, inverse_indices, loss)
        
        mean_loss = user_loss_sum.mean()
        user_total_loss = user_loss_sum[inverse_indices]
        
        eps = mean_loss / (user_total_loss + 1e-9) - 1
        return lmabda * torch.sigmoid(eps)

    def _calc_vat_loss(self, user_ids, u_g, pos_g, neg_g):
        """
        计算对抗扰动并返回对抗 Loss
        Args:
            u_g, pos_g, neg_g: 经过 GCN 传播后的 Embedding (Detached)
        """
        # 1. 复制并开启梯度 (针对 Embedding 输出，而不是模型参数)
        u_emb = u_g.detach().requires_grad_(True)
        p_emb = pos_g.detach().requires_grad_(True)
        n_emb = neg_g.detach().requires_grad_(True)

        # 2. 前向计算 Loss (Vector形式，为了计算 adaptive weight)
        loss_vec = self._calc_pure_bpr_loss(u_emb, p_emb, n_emb, reduction='none')
        
        # 3. 反向传播获取梯度
        loss_vec.mean().backward()

        # 4. 计算用户自适应权重
        user_eps = self._get_user_eps(user_ids, loss_vec, self.user_lmb)

        # 5. 生成归一化扰动 (Normalize)
        def normalize(grad):
            return self.epsilon * grad / (torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)

        u_pert = normalize(u_emb.grad)
        p_pert = normalize(p_emb.grad)
        n_pert = normalize(n_emb.grad)

        # 6. 应用自适应权重
        user_eps_view = user_eps.view(-1, 1)
        u_pert = u_pert * user_eps_view
        p_pert = p_pert * user_eps_view
        n_pert = n_pert * user_eps_view
        
        # 7. 将扰动加到原始 Embedding 上 (Detached)
        # 注意：这里我们使用原始传入的 u_g (它是计算图中保留梯度的部分) 加上 detached 的扰动
        # 这样反向传播时梯度会通过 u_g 流回 GCN 权重
        p_u_g = u_g + u_pert.detach()
        p_pos_g = pos_g + p_pert.detach()
        p_neg_g = neg_g + n_pert.detach()

        # 8. 计算对抗 Loss
        adv_loss = self._calc_pure_bpr_loss(p_u_g, p_pos_g, p_neg_g, reduction='mean')
        
        return adv_loss

    def forward(self, user_ids, pos_ids, neg_ids, decay=1e-4, user_adv=False):
        """
        Args:
            user_adv (bool): 是否开启 VAT
        """
        # 1. 索引处理
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids).long().to(self.device)
        if not isinstance(pos_ids, torch.Tensor):
            pos_ids = torch.tensor(pos_ids).long().to(self.device)
        if not isinstance(neg_ids, torch.Tensor):
            neg_ids = torch.tensor(neg_ids).long().to(self.device)

        # 2. 全图传播 (LightGCN 核心)
        all_users, all_items = self._propagate()
        
        # 3. 获取当前 batch 的 Embedding (保留计算图，用于更新模型参数)
        u_g = all_users[user_ids]
        pos_g = all_items[pos_ids]
        neg_g = all_items[neg_ids]

        # 4. 计算原始 BPR Loss
        loss = self._calc_pure_bpr_loss(u_g, pos_g, neg_g, reduction='mean')

        # 5. 计算 L2 正则 Loss (使用 Layer 0 原始 Embedding)
        u_ego = self.embedding_user(user_ids)
        pos_ego = self.embedding_item(pos_ids)
        neg_ego = self.embedding_item(neg_ids)
        reg_loss = (1/2) * (u_ego.norm(2).pow(2) + 
                             pos_ego.norm(2).pow(2) + 
                             neg_ego.norm(2).pow(2)) / float(len(user_ids))
        
        total_loss = loss + decay * reg_loss

        # 6. VAT 对抗训练部分
        if user_adv:
            # 传入当前 batch 的 Embedding 进行对抗攻击
            # 注意：VAT 通常不包含正则项，只针对 BPR 分类面进行攻击
            adv_loss = self._calc_vat_loss(user_ids, u_g, pos_g, neg_g)
            total_loss = total_loss + self.alpha * adv_loss

        return total_loss

    def predict(self, user_ids, item_ids):
        """推理接口"""
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
        """Top-K 推荐接口"""
        if not torch.is_tensor(user_ids):
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
            
        all_users, all_items = self._propagate()
        u_emb = all_users[user_ids]
        ratings = torch.matmul(u_emb, all_items.t())
        return ratings # 一般 LightGCN 输出 logit，不需要 sigmoid，除非做概率预测