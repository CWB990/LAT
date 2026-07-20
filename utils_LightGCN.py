import logging
import scipy as sp
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import dok_matrix
EPS = 1E-20


def hit_ndcg_mrr_map(ranklist, pos_item):
    """Single-positive ranking metrics."""
    if pos_item in ranklist:
        idx = ranklist.index(pos_item)
        hit = 1.0
        ndcg = 1.0 / np.log2(idx + 2)
        mrr = 1.0 / (idx + 1)
        ap = 1.0 / (idx + 1)
    else:
        hit = 0.0
        ndcg = 0.0
        mrr = 0.0
        ap = 0.0
    return hit, ndcg, mrr, ap

def evaluate_model(model, testRatings, testNegatives,
                   topK=10, topK1=20, topK2=50, topK3=100,
                   num_thread=1, device=None, batch_size=100):
    """
    适配后：调用 model(u, i) 而不是 model(u, i, False)
    """
    model.eval()
    hits, ndcgs, maps, mrrs = [], [], [], []

    INFERENCE_BATCH_SIZE = 2048 # 适当调大，MLP推理很快
    num_users = len(testRatings)

    with torch.no_grad():
        # 第一层：按用户分批处理
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)

            batch_users = []
            batch_items = []
            batch_pos_items = []
            user_slices = []
            cursor = 0

            # 准备数据
            for idx in range(start_idx, end_idx):
                user = testRatings[idx][0]
                pos_item = testRatings[idx][1]
                neg_items = testNegatives[idx]

                items = neg_items + [pos_item]
                users = [user] * len(items)

                batch_users.extend(users)
                batch_items.extend(items)
                batch_pos_items.append(pos_item)

                length = len(items)
                user_slices.append((cursor, cursor + length))
                cursor += length

            if not batch_users:
                continue

            user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
            item_tensor = torch.tensor(batch_items, dtype=torch.long, device=device)

            total_pairs = len(batch_users)
            preds_list = []
            
            # 第二层：Inference Batch 推理
            for i in range(0, total_pairs, INFERENCE_BATCH_SIZE):
                j = min(i + INFERENCE_BATCH_SIZE, total_pairs)
                u_batch = user_tensor[i:j]
                i_batch = item_tensor[i:j]
                
                # === 修改点：直接调用模型，不传 label ===
                batch_pred = model.predict(u_batch, i_batch) 
                
                preds_list.append(batch_pred.view(-1).cpu())

            all_scores_np = torch.cat(preds_list).numpy()

            # 计算指标
            for i, (s, e) in enumerate(user_slices):
                scores = all_scores_np[s:e]
                items = batch_items[s:e]
                pos_item = batch_pos_items[i]

                item_score_dict = dict(zip(items, scores))
                ranked_items = sorted(
                    item_score_dict, key=item_score_dict.get, reverse=True)[:topK]

                hit, ndcg, mrr, ap = hit_ndcg_mrr_map(ranked_items, pos_item)
                hits.append(hit)
                ndcgs.append(ndcg)
                maps.append(ap)
                mrrs.append(mrr)

    return hits, ndcgs, maps, mrrs


def calculate_target_metrics(model, target_items, target_users, unrated_items_dict,
                             topk_list=(10, 20, 50, 100), device=None, batch_size=50):
    """
    适配后：去除 model 调用时的 False 参数，并增加显存保护
    """
    # 兼容 DataParallel
    model = model.module if hasattr(model, 'module') else model
    model.eval()

    target_items = list(target_items)
    target_set = set(target_items)

    hr_results = {k: [] for k in topk_list}
    ndcg_results = {k: [] for k in topk_list}
    
    # 防止 OOM 的推理 batch size
    EVAL_BATCH_SIZE = 2048 

    with torch.no_grad():
        for user in target_users:
            if user not in unrated_items_dict:
                continue

            # 候选物品 = 未评分物品 ∪ target_items
            candidates = list(unrated_items_dict[user])
            eval_items = list(set(candidates) | target_set)

            # 构造 user-item 对
            user_tensor = torch.tensor([user] * len(eval_items), dtype=torch.long, device=device)
            item_tensor = torch.tensor(eval_items, dtype=torch.long, device=device)
            
            # === 修改点：增加 Batch 推理防止 OOM (因为 unrated items 可能非常多) ===
            scores_list = []
            num_candidates = len(eval_items)
            for i in range(0, num_candidates, EVAL_BATCH_SIZE):
                j = min(i + EVAL_BATCH_SIZE, num_candidates)
                u_batch = user_tensor[i:j]
                i_batch = item_tensor[i:j]
                
                # === 修改点：去掉 False 参数 ===
                batch_scores = model.predict(u_batch, i_batch)
                scores_list.append(batch_scores.view(-1).cpu())
                
            scores = torch.cat(scores_list).numpy()

            # 排序得到 Top-K
            max_k = max(topk_list)
            if len(scores) > max_k:
                # 使用 argpartition 加速 TopK 排序
                topk_idx = np.argpartition(scores, -max_k)[-max_k:]
                topk_scores = scores[topk_idx]
                sorted_idx = topk_idx[np.argsort(-topk_scores)]
                ranked_items = [eval_items[i] for i in sorted_idx]
            else:
                sorted_idx = np.argsort(-scores)
                ranked_items = [eval_items[i] for i in sorted_idx]

            # === 计算 HR / NDCG（user-level） ===
            for k in topk_list:
                topk_items = ranked_items[:k]

                # HR
                hit_count = sum(1 for item in target_items if item in topk_items)
                hr = hit_count / len(target_items)
                hr_results[k].append(hr)

                # NDCG
                dcg = 0.0
                idcg = 0.0
                ideal_hits = min(k, len(target_items))
                for i in range(ideal_hits):
                    idcg += 1.0 / np.log2(i + 2)

                found = 0
                for rank, item in enumerate(topk_items):
                    if item in target_set:
                        dcg += 1.0 / np.log2(rank + 2)
                        found += 1
                        if found == len(target_items):
                            break
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_results[k].append(ndcg)

    # 聚合结果
    results = {}
    for k in topk_list:
        results[f"T-HR@{k}"] = float(np.mean(hr_results[k])) if hr_results[k] else 0.0
        results[f"T-NDCG@{k}"] = float(np.mean(ndcg_results[k])) if ndcg_results[k] else 0.0

    return results

def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_adj_mat(train_data, n_users, m_items):
    """
    根据训练数据生成 LightGCN 所需的归一化拉普拉斯矩阵
    """
    print("Creating Adjacency Matrix...")
    
    # --- 修改部分开始 ---
    # 如果 train_data 是 Tensor，先转成 cpu numpy 数组
    if torch.is_tensor(train_data):
        train_data_np = train_data.cpu().numpy()
    else:
        train_data_np = train_data
        
    users = train_data_np[:, 0].astype(int)
    items = train_data_np[:, 1].astype(int)
    # --- 修改部分结束 ---
    
    # 构建邻接矩阵 A
    adj_mat = dok_matrix((n_users + m_items, n_users + m_items), dtype=np.float32)
    
    # 填充用户-物品交互
    # 这里也可以直接循环，因为 scipy 处理 numpy 数组更快
    for u, i in zip(users, items):
        adj_mat[u, n_users + i] = 1.0
        adj_mat[n_users + i, u] = 1.0
    
    adj_mat = adj_mat.tocsr()
    
    # 归一化 D^-1/2 * A * D^-1/2
    rowsum = np.array(adj_mat.sum(1))
    # 防止除以 0
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    
    norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
    norm_adj = norm_adj.tocoo()
    
    # 转为 PyTorch 稀疏张量
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)


def get_target_users_and_unrated_items(num_users, num_items, train_group, target_items):

    target_users = []
    unrated_items_dict = {}

    user_pos = {u: set() for u in range(num_users)}
    for u, i, j in train_group.tolist():
        user_pos[u].add(i)

    all_items = set(range(num_items))

    for user_id in range(num_users):
        rated_items = user_pos[user_id]
        if any(t in rated_items for t in target_items):
            continue
        target_users.append(user_id)
        unrated_items_dict[user_id] = list(all_items - rated_items)

    return target_users, unrated_items_dict



class AdvWeightPerturb_LightGCN_TWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        """
        model: 主模型
        proxy: 代理模型 (结构相同)
        proxy_optim: 代理模型的优化器 (通常用 SGD)
        gamma: 扰动幅度系数。可以是 float 或 list。
               LightGCN 只有两组主要参数 (User Emb, Item Emb)。
               如果 gamma 是 list，长度应为 2。
        """
        super(AdvWeightPerturb_LightGCN_TWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, user_ids, pos_item_ids, neg_item_ids, 
                 target_items, alpha_awp, device):
        """
        计算对抗扰动 (Gradient Ascent on Proxy)
        """
        # 1. 同步参数
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        # -------- Non-Target (Clean) Loss Weighting --------
        # 计算每个样本的 loss (batch_size 维度)
        individual_losses = self.proxy.get_individual_loss(user_ids, pos_item_ids, neg_item_ids)
        individual_losses = individual_losses.view(-1)

        # 获取 Batch 中唯一用户
        unique_users, inverse_indices = torch.unique(user_ids, return_inverse=True)
        num_unique = unique_users.size(0)
        
        # 聚合每个用户的 Total Loss
        user_loss_sum = torch.zeros(num_unique, dtype=individual_losses.dtype, device=device)
        user_loss_sum.scatter_add_(0, inverse_indices, individual_losses.detach())

        # 计算权重 (Difficulty aware)
        mean_loss = user_loss_sum.mean()
        user_total_loss = user_loss_sum[inverse_indices] # 映射回 batch 维度
        
        # 这里的逻辑是：Loss 越大的用户，eps 越小 -> sigmoid 后权重越小 (关注简单样本? 或者反过来)
        # 根据 TWP 论文，通常给难样本降权，或者是为了提升鲁棒性给易受攻击用户加权
        # 这里的实现逻辑是：loss 大 -> eps 小 -> weight 小 (Long-tail 保护)
        eps_weight = mean_loss / (user_total_loss + 1e-9) - 1
        loss_weights = torch.sigmoid(eps_weight)

        # 加权后的 Clean Loss
        non_target_loss = (individual_losses * loss_weights).sum()

        # -------- Target Item Attack Simulation --------
        # 目的是最大化目标物品的排名，即最小化 (target - neg) 的 BPR Loss
        # 但在 AWP 中，我们是 Attack，所以我们要让这个 Loss 变大 (让模型在扰动下 推荐 Target 变差)
        # 或者是：我们通过扰动让模型把 Target 推荐出来 (Enhance Target) ?
        # TWP 的逻辑是：寻找一种扰动，使得 Clean Loss 变大 且 Target Loss 也变大 (Model 变差)。
        # 然后通过最小化这个扰动后的 Loss 来训练 Robust Model。
        
        if len(target_items) > 0:
            target_items_tensor = torch.tensor(target_items, device=device)
            num_targets = len(target_items_tensor)
            
            # 这里的策略：让当前 Batch 的 User 去和 Target Items 产生交互
            # 为了内存考虑，如果 unique_users 太多，可以采样
            target_users_exp = unique_users.repeat_interleave(num_targets)
            pos_items_exp = target_items_tensor.repeat(num_unique)
            
            # 随机采样负样本
            rand_neg = torch.randint(0, self.model.item_count, size=pos_items_exp.shape, device=device)
            # 简单去重：如果负样本刚好是 target，就随机换一个 (简化处理，概率很低)
            
            # 计算 Target Loss (不加权)
            target_loss_vec = self.proxy.get_individual_loss(target_users_exp, pos_items_exp, rand_neg)
            target_loss = target_loss_vec.sum()
        else:
            target_loss = 0.0

        # -------- AWP Objective (Gradient Ascent) --------
        # 我们寻找参数 W' 使得 Loss 最大化
        # Loss = (1 - alpha) * L_clean + alpha * L_target
        loss = -1.0 * ((1.0 - alpha_awp) * non_target_loss + alpha_awp * target_loss)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # 计算扰动差值
        diff = diff_in_weights_LightGCN(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        # LightGCN 的参数主要是 Embeddings，都在 state_dict 里
        # 如果 gamma 是 float，扩展为 list
        gamma_list = self.gamma if isinstance(self.gamma, list) else [self.gamma] * len(diff)
        
        add_into_weights_LightGCN(self.model, diff, coeff=gamma_list)

    def restore(self, diff):
        gamma_list = self.gamma if isinstance(self.gamma, list) else [self.gamma] * len(diff)
        # 恢复时系数取反
        gamma_list = [-1.0 * g for g in gamma_list]
        add_into_weights_LightGCN(self.model, diff, coeff=gamma_list)

# --- 辅助函数 ---
def diff_in_weights_LightGCN(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        # LightGCN 的权重主要是 embedding.weight，是 2 维的
        if len(old_w.size()) <= 1: 
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            # 归一化扰动
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict

def add_into_weights_LightGCN(model, diff, coeff):
    names_in_diff = list(diff.keys())
    # 确保 coeff 长度足够，或者循环使用
    if len(coeff) < len(names_in_diff):
        coeff = coeff * (len(names_in_diff) // len(coeff) + 1)
    
    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff[i] * diff[name])
                i += 1


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


# def add_into_weights(model, diff, seeds, coeff=1.0):  # 使用pgd更新模型参数
#     names_in_diff = diff.keys()
#     i = 0
#     with torch.no_grad():
#         for name, param in model.named_parameters():
#             if name in names_in_diff:
#                 if seeds[i] == 1:
#                     param.add_(coeff * diff[name])
#                 i = i+1

def add_into_weights(model, diff, seeds, coeff=1.0):
    names_in_diff = diff.keys()
    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                # 循环利用 seeds，防止参数数量超过 seeds 长度
                seed_val = seeds[i % len(seeds)] 
                if seed_val == 1:
                    param.add_(coeff * diff[name])
                i += 1


class AdvWeightPerturb_LightGCN_RAWP(object):

    def __init__(self, model, proxy, proxy_optim, gamma, device):
        super(AdvWeightPerturb_LightGCN_RAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.device = device

    def calc_awp(self, user_ids, pos_item_ids, neg_item_ids):
        # 1. 复制当前参数到 Proxy
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        self.proxy_optim.zero_grad()
        
        # 2. 计算 Proxy 的 Loss
        # 注意：这里 decay=0.0，只最大化 BPR Loss，不包括正则项
        # 我们希望找到让 BPR Loss 变大（预测变差）的方向，所以取负号
        loss = - self.proxy(user_ids, pos_item_ids, neg_item_ids, decay=0.0)
        
        loss.backward()
        self.proxy_optim.step()
        
        # 3. 计算扰动幅度
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff, seeds):
        add_into_weights(self.model, diff, seeds, coeff=1.0 * self.gamma)

    def restore(self, diff, seeds):
        add_into_weights(self.model, diff, seeds, coeff=-1.0 * self.gamma)


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        return torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()