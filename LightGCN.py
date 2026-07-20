import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, user_count, item_count, device, adj_mat, latent_dim=64, n_layers=3, keep_prob=1.0):
        super(LightGCN, self).__init__()

        # 基础属性
        self.user_count = user_count
        self.item_count = item_count
        self.device = device
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.keep_prob = keep_prob
        
        # 图结构 (稀疏矩阵)
        self.Graph = adj_mat.to(device)

        # Embedding 层 (对应 ego embeddings)
        self.embedding_user = nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.latent_dim)

        # 初始化：LightGCN 通常使用 Normal 初始化
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def _dropout_graph(self, graph, keep_prob):
        """对稀疏图进行 Dropout"""
        size = graph.size()
        index = graph.indices().t()
        values = graph.values()
        random_index = torch.rand(len(values), device=self.device) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        return torch.sparse.FloatTensor(index.t(), values, size).to(self.device)

    def _propagate(self):
        """核心图卷积过程 (对应原版的 computer)"""
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        
        # 训练时考虑 Dropout
        if self.training and self.keep_prob < 1.0:
            g_droped = self._dropout_graph(self.Graph, self.keep_prob)
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        
        # 均值聚合各层 Embedding
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.user_count, self.item_count])
        return users, items

    def _calc_bpr_loss(self, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0):
        """计算 BPR Loss 和 L2 正则"""
        # 1. 计算 BPR 损失 (softplus 相当于 log(1 + exp(neg-pos)))
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # 2. 计算 L2 正则项 (使用的是原始 Embedding)
        reg_loss = (1/2) * (userEmb0.norm(2).pow(2) + 
                             posEmb0.norm(2).pow(2) + 
                             negEmb0.norm(2).pow(2)) / float(len(users_emb))
        
        return loss, reg_loss

    def forward(self, user_ids, pos_ids, neg_ids, decay=1e-4):
        """
        兼容 ConvNCF 格式的 forward
        user_ids, pos_ids, neg_ids: 均为 LongTensor
        decay: 正则化权重 (lambda)
        """
        # 1. 确保 ID 为 long 类型并移动到设备
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(user_ids).long().to(self.device)
        if not isinstance(pos_ids, torch.Tensor):
            pos_ids = torch.tensor(pos_ids).long().to(self.device)
        if not isinstance(neg_ids, torch.Tensor):
            neg_ids = torch.tensor(neg_ids).long().to(self.device)

        # 2. 获取经由 GCN 传播后的 Embedding
        all_users, all_items = self._propagate()
        
        u_g_embeddings = all_users[user_ids]
        pos_i_g_embeddings = all_items[pos_ids]
        neg_i_g_embeddings = all_items[neg_ids]

        # 3. 获取原始 Embedding 用于正则化
        u_ego_embeddings = self.embedding_user(user_ids)
        pos_ego_embeddings = self.embedding_item(pos_ids)
        neg_ego_embeddings = self.embedding_item(neg_ids)

        # 4. 计算 Loss
        bpr_loss, reg_loss = self._calc_bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings,
            u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings
        )
        
        # 返回总 Loss
        return bpr_loss + decay * reg_loss

    def predict(self, user_ids, item_ids):
        """
        推理/测试接口
        返回指定用户对指定物品的分数
        """
        if not torch.is_tensor(user_ids):
            user_ids = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        if not torch.is_tensor(item_ids):
            item_ids = torch.tensor(item_ids, dtype=torch.long, device=self.device)

        # 获取传播后的 Embedding
        all_users, all_items = self._propagate()
        
        u_emb = all_users[user_ids]
        i_emb = all_items[item_ids]
        
        # 计算内积得分
        prediction = torch.sum(torch.mul(u_emb, i_emb), dim=1)
        return prediction

    def get_all_ratings(self, user_ids):
        """
        用于 Top-K 推荐：一次性返回指定用户对所有物品的预测分数
        """
        all_users, all_items = self._propagate()
        u_emb = all_users[user_ids.long()]
        # (users, latent) @ (latent, all_items) -> (users, all_items)
        ratings = torch.matmul(u_emb, all_items.t())
        return torch.sigmoid(ratings)