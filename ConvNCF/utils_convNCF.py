import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch
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
                   num_thread=1, device='cuda', batch_size=100):
    """OOM-safe evaluation (reused from your current training script)."""
    model.eval()
    hits, ndcgs, maps, mrrs = [], [], [], []

    INFERENCE_BATCH_SIZE = 1024
    num_users = len(testRatings)

    with torch.no_grad():
        # 第一层：按用户分批处理,每次处理batch_size个用户
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)

            batch_users = []
            batch_items = []
            batch_pos_items = []
            user_slices = []
            cursor = 0

            for idx in range(start_idx, end_idx):
                len(testNegatives[idx])
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

            user_tensor = torch.tensor(
                batch_users, dtype=torch.long, device=device)
            item_tensor = torch.tensor(
                batch_items, dtype=torch.long, device=device)

            total_pairs = len(batch_users)
            preds_list = []
            # 第二层：在用户批次内按推理批次处理，每个用户批次内的用户-物品对再按INFERENCE_BATCH_SIZE=4096分批推理
            for i in range(0, total_pairs, INFERENCE_BATCH_SIZE):
                j = min(i + INFERENCE_BATCH_SIZE, total_pairs)
                u_batch = user_tensor[i:j]
                i_batch = item_tensor[i:j]
                batch_pred = model(u_batch, i_batch, False).view(-1).cpu()
                preds_list.append(batch_pred)

            all_scores_np = torch.cat(preds_list).numpy()

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
                             topk_list=(10, 20, 50, 100), device='cuda', batch_size=50):
    """
    User-level target HR / NDCG (legacy version).
    Interface is kept unchanged for fair comparison with other models.
    """

    # 兼容 DataParallel
    model = model.module if hasattr(model, 'module') else model
    model.eval()

    target_items = list(target_items)
    target_set = set(target_items)

    hr_results = {k: [] for k in topk_list}
    ndcg_results = {k: [] for k in topk_list}

    with torch.no_grad():
        for user in target_users:
            if user not in unrated_items_dict:
                continue

            # 候选物品 = 未评分物品 ∪ target_items（防止 target item 被过滤）
            candidates = list(unrated_items_dict[user])
            eval_items = list(set(candidates) | target_set)

            # 构造 user-item 对
            user_tensor = torch.tensor([user] * len(eval_items),
                                       dtype=torch.long, device=device)
            item_tensor = torch.tensor(eval_items,
                                       dtype=torch.long, device=device)

            # 前向打分
            scores = model(user_tensor, item_tensor, False) \
                .view(-1).detach().cpu().numpy()

            # 排序得到 Top-K
            max_k = max(topk_list)
            if len(scores) > max_k:
                topk_idx = np.argpartition(scores, -max_k)[-max_k:]
                topk_scores = scores[topk_idx]
                sorted_idx = topk_idx[np.argsort(-topk_scores)]
            else:
                sorted_idx = np.argsort(-scores)

            ranked_items = [eval_items[i] for i in sorted_idx]

            # === 计算 HR / NDCG（user-level） ===
            for k in topk_list:
                topk_items = ranked_items[:k]

                # HR
                hit_count = sum(
                    1 for item in target_items if item in topk_items)
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
        results[f"T-HR@{k}"] = float(np.mean(hr_results[k])
                                     ) if hr_results[k] else 0.0
        results[f"T-NDCG@{k}"] = float(np.mean(ndcg_results[k])
                                       ) if ndcg_results[k] else 0.0

    return results


def evaluate_model_apr(model, testRatings, testNegatives,
                       topK=10, topK1=20, topK2=50, topK3=100,
                       num_thread=1, device='cuda', batch_size=100, is_pretrain_eval=False):
    """OOM-safe evaluation (reused from your current training script)."""
    model.eval()
    hits, ndcgs, maps, mrrs = [], [], [], []

    INFERENCE_BATCH_SIZE = 4096
    num_users = len(testRatings)

    with torch.no_grad():
        # 第一层：按用户分批处理,每次处理batch_size个用户
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)

            batch_users = []
            batch_items = []
            batch_pos_items = []
            user_slices = []
            cursor = 0

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

            user_tensor = torch.tensor(
                batch_users, dtype=torch.long, device=device)
            item_tensor = torch.tensor(
                batch_items, dtype=torch.long, device=device)

            total_pairs = len(batch_users)
            preds_list = []
            # 第二层：在用户批次内按推理批次处理，每个用户批次内的用户-物品对再按INFERENCE_BATCH_SIZE=4096分批推理
            for i in range(0, total_pairs, INFERENCE_BATCH_SIZE):
                j = min(i + INFERENCE_BATCH_SIZE, total_pairs)
                u_batch = user_tensor[i:j]
                i_batch = item_tensor[i:j]
                batch_pred = model.predict(u_batch, i_batch, is_pretrain=is_pretrain_eval).view(-1).cpu()
                preds_list.append(batch_pred)

            all_scores_np = torch.cat(preds_list).numpy()

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


def calculate_target_metrics_apr(model, target_items, target_users, unrated_items_dict,
                                 topk_list=(10, 20, 50, 100), device='cuda', batch_size=50, is_pretrain_eval=False):
    """
    User-level target HR / NDCG (legacy version).
    Interface is kept unchanged for fair comparison with other models.
    """

    # 兼容 DataParallel
    model = model.module if hasattr(model, 'module') else model
    model.eval()

    target_items = list(target_items)
    target_set = set(target_items)

    hr_results = {k: [] for k in topk_list}
    ndcg_results = {k: [] for k in topk_list}

    with torch.no_grad():
        for user in target_users:
            if user not in unrated_items_dict:
                continue

            # 候选物品 = 未评分物品 ∪ target_items（防止 target item 被过滤）
            candidates = list(unrated_items_dict[user])
            eval_items = list(set(candidates) | target_set)

            # 构造 user-item 对
            user_tensor = torch.tensor([user] * len(eval_items),
                                       dtype=torch.long, device=device)
            item_tensor = torch.tensor(eval_items,
                                       dtype=torch.long, device=device)

            # 前向打分
            scores = model.predict(user_tensor, item_tensor, is_pretrain=is_pretrain_eval).view(-1).detach().cpu().numpy()

            # 排序得到 Top-K
            max_k = max(topk_list)
            if len(scores) > max_k:
                topk_idx = np.argpartition(scores, -max_k)[-max_k:]
                topk_scores = scores[topk_idx]
                sorted_idx = topk_idx[np.argsort(-topk_scores)]
            else:
                sorted_idx = np.argsort(-scores)

            ranked_items = [eval_items[i] for i in sorted_idx]

            # === 计算 HR / NDCG（user-level） ===
            for k in topk_list:
                topk_items = ranked_items[:k]

                # HR
                hit_count = sum(
                    1 for item in target_items if item in topk_items)
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
        results[f"T-HR@{k}"] = float(np.mean(hr_results[k])
                                     ) if hr_results[k] else 0.0
        results[f"T-NDCG@{k}"] = float(np.mean(ndcg_results[k])
                                       ) if ndcg_results[k] else 0.0

    return results


def evaluate_model_vat(model, testRatings, testNegatives,
                   topK=10, topK1=20, topK2=50, topK3=100,
                   num_thread=1, device='cuda', batch_size=100,is_pretrain_eval=False):
    """OOM-safe evaluation (reused from your current training script)."""
    model.eval()
    hits, ndcgs, maps, mrrs = [], [], [], []

    INFERENCE_BATCH_SIZE = 1024
    num_users = len(testRatings)

    with torch.no_grad():
        # 第一层：按用户分批处理,每次处理batch_size个用户
        for start_idx in range(0, num_users, batch_size):
            end_idx = min(start_idx + batch_size, num_users)

            batch_users = []
            batch_items = []
            batch_pos_items = []
            user_slices = []
            cursor = 0

            for idx in range(start_idx, end_idx):
                len(testNegatives[idx])
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

            user_tensor = torch.tensor(
                batch_users, dtype=torch.long, device=device)
            item_tensor = torch.tensor(
                batch_items, dtype=torch.long, device=device)

            total_pairs = len(batch_users)
            preds_list = []
            # 第二层：在用户批次内按推理批次处理，每个用户批次内的用户-物品对再按INFERENCE_BATCH_SIZE=4096分批推理
            for i in range(0, total_pairs, INFERENCE_BATCH_SIZE):
                j = min(i + INFERENCE_BATCH_SIZE, total_pairs)
                u_batch = user_tensor[i:j]
                i_batch = item_tensor[i:j]
                batch_pred = model.predict(u_batch, i_batch, is_pretrain=is_pretrain_eval).view(-1).cpu()
                preds_list.append(batch_pred)

            all_scores_np = torch.cat(preds_list).numpy()

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


def calculate_target_metrics_vat(model, target_items, target_users, unrated_items_dict,
                             topk_list=(10, 20, 50, 100), device='cuda', batch_size=50,is_pretrain_eval=False):
    """
    User-level target HR / NDCG (legacy version).
    Interface is kept unchanged for fair comparison with other models.
    """

    # 兼容 DataParallel
    model = model.module if hasattr(model, 'module') else model
    model.eval()

    target_items = list(target_items)
    target_set = set(target_items)

    hr_results = {k: [] for k in topk_list}
    ndcg_results = {k: [] for k in topk_list}

    with torch.no_grad():
        for user in target_users:
            if user not in unrated_items_dict:
                continue

            # 候选物品 = 未评分物品 ∪ target_items（防止 target item 被过滤）
            candidates = list(unrated_items_dict[user])
            eval_items = list(set(candidates) | target_set)

            # 构造 user-item 对
            user_tensor = torch.tensor([user] * len(eval_items),
                                       dtype=torch.long, device=device)
            item_tensor = torch.tensor(eval_items,
                                       dtype=torch.long, device=device)

            # 前向打分
            scores = model.predict(user_tensor, item_tensor, is_pretrain=is_pretrain_eval) \
                .view(-1).detach().cpu().numpy()

            # 排序得到 Top-K
            max_k = max(topk_list)
            if len(scores) > max_k:
                topk_idx = np.argpartition(scores, -max_k)[-max_k:]
                topk_scores = scores[topk_idx]
                sorted_idx = topk_idx[np.argsort(-topk_scores)]
            else:
                sorted_idx = np.argsort(-scores)
            ranked_items = [eval_items[i] for i in sorted_idx]

            # === 计算 HR / NDCG（user-level） ===
            for k in topk_list:
                topk_items = ranked_items[:k]

                # HR
                hit_count = sum(
                    1 for item in target_items if item in topk_items)
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
        results[f"T-HR@{k}"] = float(np.mean(hr_results[k])
                                     ) if hr_results[k] else 0.0
        results[f"T-NDCG@{k}"] = float(np.mean(ndcg_results[k])
                                       ) if ndcg_results[k] else 0.0

    return results


class AdvWeightPerturb_convNCF_RAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma, device):
        super(AdvWeightPerturb_convNCF_RAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        self.device = device

    def calc_awp(self, user_ids, pos_item_ids, neg_item_ids):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        bpr_loss = BPRLoss().to(self.device)
        self.proxy_optim.zero_grad()
        pos_preds = self.proxy(user_ids, pos_item_ids, False)
        neg_preds = self.proxy(user_ids, neg_item_ids, False)
        loss = - bpr_loss(pos_preds, neg_preds)
        loss.backward()
        self.proxy_optim.step()
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff, seeds):
        add_into_weights(self.model, diff, seeds, coeff=1.0 * self.gamma)

    def restore(self, diff, seeds):
        add_into_weights(self.model, diff, seeds, coeff=-1.0 * self.gamma)


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


def add_into_weights(model, diff, seeds, coeff=1.0):  # 使用pgd更新模型参数
    names_in_diff = diff.keys()
    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                if seeds[i] == 1:
                    param.add_(coeff * diff[name])
                i = i+1


def add_into_weights_con(model, diff, coeff, ratio=0.4):  # 使用pgd更新模型参数
    names_in_diff = diff.keys()
    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff[i] * diff[name])
                i = i + 1


def add_into_weights(model, diff, seeds, coeff=1.0, ratio=0.5):  # 使用pgd更新模型参数
    names_in_diff = diff.keys()
    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                if seeds[i] == 1:
                    param.add_(coeff * diff[name])


class AdvWeightPerturb_ConvNCF_TWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb_ConvNCF_TWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, args, user_ids, pos_item_ids, neg_item_ids,
             target_items, alpha_awp):

        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        # -------- BPR forward --------
        # 1. 获取 loss，并确保它是 1D 的 [batch_size]
        # 如果模型输出是 [batch_size, 1]，会导致 scatter 报错，必须 view(-1)
        individual_losses = self.proxy(user_ids, pos_item_ids, neg_item_ids, False, 'none')
        individual_losses = individual_losses.view(-1) 

        # -------- user-level weighting --------
        # 2. 获取唯一用户和反向索引
        # unique_users: [num_unique_users]
        # inverse_indices: [batch_size]，值域为 [0, num_unique_users - 1]
        unique_users, inverse_indices = torch.unique(
            user_ids, return_inverse=True)

        # 3. 初始化累加容器
        # 大小直接设为 len(unique_users)，这是绝对安全的，不需要计算 max_index
        num_unique = unique_users.size(0)
        user_loss_sum = torch.zeros(num_unique, dtype=individual_losses.dtype, device=unique_users.device)

        # 4. 执行 scatter_add
        # index: [batch_size] -> inverse_indices
        # src:   [batch_size] -> individual_losses
        # self:  [num_unique] -> user_loss_sum
        # 逻辑：将属于同一个 user 的 loss 加在一起
        user_loss_sum.scatter_add_(0, inverse_indices, individual_losses.detach())

        # 5. 计算权重
        # mean_loss: 所有唯一用户的平均总 loss
        mean_loss = user_loss_sum.mean()
        
        # user_total_loss: 映射回 batch 维度
        # 直接使用 inverse_indices 取值即可，绝对不会越界
        # 这一步得到 [batch_size] 大小的 tensor，每个位置是该样本所属用户的总 loss
        user_total_loss = user_loss_sum[inverse_indices]

        # 避免除以 0
        eps = mean_loss / (user_total_loss + 1e-9) - 1
        loss_weights = torch.sigmoid(eps)

        # 计算最终加权 loss
        non_target_loss = (individual_losses * loss_weights).sum()

        # -------- target item BPR loss --------

        target_items_tensor = torch.tensor(target_items, device=args.device)
        num_users = len(unique_users)
        num_targets = len(target_items_tensor)
        target_users = unique_users.repeat_interleave(num_targets)
        pos_items = target_items_tensor.repeat(num_users)

        # ========= negative sampling =========
        # 所有候选 item id：0,1,...,num_items-1
        all_items = torch.arange(args.nItems, device=args.device)
        # 去掉所有 target_items
        mask = torch.ones_like(all_items, dtype=torch.bool)
        mask[target_items_tensor] = False
        candidate_items = all_items[mask]          # 这里不包含任何 target item
        # 从 candidate_items 中按均匀分布采负样本
        rand_idx = torch.randint(
            low=0,
            high=candidate_items.size(0),
            size=pos_items.shape,
            device=args.device
        )
        neg_items = candidate_items[rand_idx]

        target_loss = self.proxy(target_users, pos_items, neg_items, False)

        # -------- AWP objective --------
        loss = -((1 - alpha_awp) * non_target_loss + alpha_awp * target_loss)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        # 对序列中的每个元素乘以 1.0
        new_coeff = [1.0 * g for g in self.gamma]
        add_into_weights_con(self.model, diff, coeff=new_coeff)

    def restore(self, diff):
        # 对序列中的每个元素乘以 -1.0
        new_coeff = [-1.0 * g for g in self.gamma]
        add_into_weights_con(self.model, diff, coeff=new_coeff)


class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        return torch.nn.functional.softplus(-(pos_scores - neg_scores)).mean()


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



def evaluate_model_TWP(model, testRatings, testNegatives,
                   topK=10, topK1=20, topK2=50, topK3=100,
                   num_thread=1, device='cuda', batch_size=100):
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
                batch_pred = model.predict(u_batch, i_batch, False) 
                
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


def calculate_target_metrics_TWP(model, target_items, target_users, unrated_items_dict,
                             topk_list=(10, 20, 50, 100), device='cuda', batch_size=50):
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
