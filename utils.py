from copy import deepcopy
import heapq
import logging
import math
import multiprocessing
import time
import numpy as np
from collections import namedtuple
import torch
from torch import nn
# import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import pandas as pd
from scipy import sparse
import numpy as np
import scipy.sparse as sp

def cal_lp_norm(tensor,p,dim_count):
    tmp = tensor
    for i in range(1,dim_count):
        tmp = torch.norm(tmp,p=p,dim=i,keepdim=True) #torch.Size([100, 1])
    
    
    return torch.clamp_min(tmp, 1e-8)

# evaluate

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
_K1 = None
_K2 = None
_K3 = None

def evaluate_model(model, testRatings, testNegatives, K, K1, K2,K3, num_thread, device):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    global _K1
    global _K2
    global _K3
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _K1 = K1
    _K2 = K2
    _K3 = K3
        
    hits10, ndcgs10,maps10, mrrs10 = [], [], [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr10, ndcg10, ap10, mrr10) = eval_one_rating(idx, device)
        hits10.append(hr10)
        ndcgs10.append(ndcg10)
        maps10.append(ap10)
        mrrs10.append(mrr10)
     
    return (hits10, ndcgs10, maps10, mrrs10)


def eval_one_rating(idx, device):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    r = rating[2]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int32')
    label = np.full(len(items), 0, dtype='int32')
    # ---
    dst = TestDataset_adv(users, items, label)
    ldr = torch.utils.data.DataLoader(dst, batch_size=100, shuffle=False)

    _model.eval()
    predictions = [None] * len(dst)
    total = 0
    with torch.no_grad():
        for ui, ii, lbl in ldr:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
            bsz = ui.size(0)
            _,_,_,_,_,_,_, ri = _model(ui, ii, lbl)
            # _,ri = _model(ui, ii)
            #ri = criterion(output,lbl)  
            ri = ri.squeeze().cpu().tolist()
            predictions[total:total+bsz] = ri
    # predictions = _model.predict([users, np.array(items)], 
    #                              batch_size=100, verbose=0)
    # ---
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist10 = heapq.nlargest(_K, map_item_score, key=map_item_score.get)


    hr10 = getHitRatio(ranklist10, gtItem)
    ndcg10 = getNDCG(ranklist10, gtItem)
    ap10 = getAP(ranklist10, gtItem)
    mrr10 = getMRR(ranklist10, gtItem)
 
    return (hr10, ndcg10, ap10, mrr10)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def getMRR(ranklist, gtItem):
    for index, item in enumerate(ranklist):
        if item == gtItem:
            return 1.0 / (index + 1.0)
    return 0

def getAP(ranklist, gtItem):
    hits = 0
    sum_precs = 0
    for n in range(len(ranklist)):
        if ranklist[n] == gtItem:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / 1
    else:
        return 0



################################################################
## Components from https://github.com/davidcpage/cifar10-fast ##
################################################################

#####################
## data preprocessing
#####################

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 



def get_train_instances(train, nNeg):
    import scipy.sparse as sp
    userInput, itemInput, labels = [], [], []
    nUsers, nItems = train.shape

    if isinstance(train, torch.Tensor):
        # 获取非零元素的索引
        nonzero_indices = torch.nonzero(train, as_tuple=True)
        for u, i in zip(*nonzero_indices):
            u = u.item()
            i = i.item()
            # positive instance
            userInput.append(u)
            itemInput.append(i)
            labels.append(1)
            # negative instances
            for _ in range(nNeg):
                j = np.random.randint(nItems)
                while train[u, j] != 0:
                    j = np.random.randint(nItems)
                userInput.append(u)
                itemInput.append(j)
                labels.append(0)
    else:
        # 将稀疏矩阵转换为CSR格式以支持高效的行访问
        if sp.issparse(train):
            train_csr = train.tocsr()
            
            # 获取非零元素的索引
            nonzero = train.nonzero()
            pos_u = nonzero[0]
            pos_i = nonzero[1]
            
            for idx in range(len(pos_u)):
                u = pos_u[idx]
                i = pos_i[idx]
                
                # positive instance
                userInput.append(u)
                itemInput.append(i)
                labels.append(1)
                
                # negative instances
                for _ in range(nNeg):
                    j = np.random.randint(nItems)
                    # 使用 CSR 矩阵的 getrow 方法来检查行中的元素
                    while train_csr[u].toarray().flatten()[j] != 0:
                        j = np.random.randint(nItems)
                    userInput.append(u)
                    itemInput.append(j)
                    labels.append(0)
    return userInput, itemInput, labels

def get_train_matrix(train):
    if isinstance(train, sp.dok_matrix):
        # 如果是 dok_matrix 类型，直接使用 keys 方法
        nUsers, nItems = train.shape
        trainMatrix = np.zeros([nUsers, nItems], dtype=np.int32)
        for (u, i) in train.keys():
            trainMatrix[u][i] = 1
        return trainMatrix
    elif isinstance(train, (sp.csr_matrix, sp.csc_matrix, sp.lil_matrix, sp.coo_matrix)):
        # 如果是其他稀疏矩阵类型，转换为 dok_matrix 再处理
        dok_train = train.todok()
        nUsers, nItems = dok_train.shape
        trainMatrix = np.zeros([nUsers, nItems], dtype=np.int32)
        for (u, i) in dok_train.keys():
            trainMatrix[u][i] = 1
        return trainMatrix
    elif isinstance(train, np.ndarray):
        # 如果是 numpy 数组，直接转换
        return (train > 0).astype(np.int32)
    else:
        raise ValueError(f"Unsupported train type: {type(train)}")



#####################
## data augmentation
#####################



class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:,y0:y0+self.h,x0:x0+self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)}
    
    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)
    
class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x 
        
    def options(self, x_shape):
        return {'choice': [True, False]}

class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        x = x.copy()
        x[:,y0:y0+self.h,x0:x0+self.w].fill(0.0)
        return x

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W+1-self.w), 'y0': range(H+1-self.h)} 
    
    
class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k,v) in choices.items()}
            data = f(data, **args)
        return data, labels
    
    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k:np.random.choice(v, size=N) for (k,v) in options.items()})

#####################
## dataset
#####################


#####################
## data loading
#####################

class BatchDataset(torch.utils.data.Dataset):
    def __init__(self, userInput, itemInput, labels):
        super(BatchDataset, self).__init__()
        self.userInput = torch.Tensor(userInput).long()
        self.itemInput = torch.Tensor(itemInput).long()
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        return self.userInput[index], self.itemInput[index], self.labels[index]

    def __len__(self):
        return self.labels.size(0)


class AverageMeter(object):
    """Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, fmt=".4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f"{self.name} {self.val:{self.fmt}} ({self.avg:{self.fmt}})"
        return fmtstr


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )
    
    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices() 
        return ({'input': x.to(device).half(), 'target': y.to(device).long()} for (x,y) in self.dataloader)
    
    def __len__(self): 
        return len(self.dataloader)
    
    
    
class TestDataset_adv(torch.utils.data.Dataset):
    def __init__(self, userInput, itemInput, labels):
        super(TestDataset_adv, self).__init__()
        self.userInput = torch.Tensor(userInput).long()
        self.itemInput = torch.Tensor(itemInput).long()
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        return self.userInput[index], self.itemInput[index], self.labels[index]

    def __len__(self):
        return self.userInput.size(0)
    


    
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

def create_scheduler(args, optimizer, lr_decays=None):
    
	if args.lr_scheduler == "step":
		if lr_decays is None:
			lr_decays = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decays, gamma=args.lr_decay_gamma, last_epoch=-1)
	elif args.lr_scheduler == "cosine":
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
	else:
		raise ValueError("The scheduler is not implemented!")
	# elif args.lr_scheduler == "cyclic":
	# 	pass
	return scheduler





# 测试Target items的各种指标

# def evaluate_target_items(model, testRatings, testNegatives, K, target_items, device):
#     """
#     评估目标项目在推荐列表中的命中率(HR)和NDCG
    
#     参数:
#     - model: 要评估的模型
#     - testRatings: 测试集评分数据
#     - testNegatives: 测试集负样本数据
#     - K: 推荐列表长度
#     - target_items: 要评估的目标项目列表
#     - device: 计算设备
    
#     返回:
#     - target_hr: 目标项目的命中率
#     - target_ndcg: 目标项目的NDCG值
#     """
#     target_hits = []
#     target_ndcgs = []
    
#     # 遍历所有测试样本
#     for idx in range(len(testRatings)):
#         rating = testRatings[idx]
#         items = testNegatives[idx].copy()  # 复制负样本列表
#         u = rating[0]  # 用户ID
#         gtItem = rating[1]  # 真实项目ID
        
#         # 将真实项目添加到评估列表中
#         items.append(gtItem)
        
#         # 准备测试数据
#         users = np.full(len(items), u, dtype='int32')
#         labels = np.zeros(len(items), dtype='int32')
        
#         # 创建测试数据集和数据加载器
#         dst = TestDataset_adv(users, items, labels)
#         ldr = torch.utils.data.DataLoader(dst, batch_size=100, shuffle=False)
        
#         # 模型预测
#         model.eval()
#         predictions = []
#         with torch.no_grad():
#             for ui, ii, lbl in ldr:
#                 ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
#                 # 根据模型输出调整此处的返回值
#                 _, _, _, _, _, _, _, ri = model(ui, ii, lbl)
#                 predictions.extend(ri.squeeze().cpu().tolist())
        
#         # 创建项目-分数映射
#         map_item_score = {items[i]: predictions[i] for i in range(len(items))}
        
#         # 获取Top-K推荐列表
#         ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        
#         hr10 = getHitRatio_Target(ranklist, target_items)
#         ndcg10 = getNDCG_Target(ranklist, target_items)
#         target_hits.append(hr10)
#         target_ndcgs.append(ndcg10)
    
#     # 计算平均HR和NDCG
#     target_hr = np.mean(target_hits)
#     target_ndcg = np.mean(target_ndcgs)
    
#     return target_hr, target_ndcg

# def getHitRatio_Target(ranklist, target_items):
#     for T_item in target_items:
#         for item in ranklist:
#             if item == T_item:
#                 return 1
#     return 0


# def getNDCG_Target(ranklist, target_items):
#     for T_item in target_items:
#         for i in range(len(ranklist)):
#             item = ranklist[i]
#             if item == T_item:
#                 return math.log(2) / math.log(i+2)
#     return 0




def calculate_target_metrics( model,target_items, target_users,unrate_items_dict, topk_list=[10, 20, 50, 100]):
    """
    计算目标物品在前k个推荐中的平均命中率和NDCG

    Args:
        target_users: 目标用户列表
        target_items: 目标物品列表
        model: 推荐模型
        unrate_items_dict: 用户未评分物品字典
        topk_list: 要计算的topk值列表

    Returns:
        dict: 包含各指标结果的字典
    """
    # 将target_items转换为集合用于快速查找
    model = model.module if hasattr(model, 'module') else model

    target_set = set(target_items)

    # 初始化结果存储
    hr_results = {k: [] for k in topk_list}
    ndcg_results = {k: [] for k in topk_list}

    for user in target_users:
        # 获取推荐结果
        recommended_items, recommended_scores = model.recommend(user, unrate_items_dict[user], topk=max(topk_list))

        # 创建位置映射字典
        item_to_rank = {item: rank for rank, item in enumerate(recommended_items)}

        for k in topk_list:
            # 计算命中率
            hit_count = 0
            for item in target_items:
                if item in recommended_items[:k]:
                    hit_count += 1

            hr = hit_count / len(target_items)
            hr_results[k].append(hr)

            # 计算NDCG
            dcg = 0.0
            idcg = 0.0

            # 计算理想DCG（前min(k, len(target_items))个相关物品）
            ideal_ranks = min(k, len(target_items))
            for i in range(ideal_ranks):
                idcg += 1.0 / np.log2(i + 2)  # i+2因为排名从1开始

            # 计算实际DCG
            found_targets = 0
            for i, item in enumerate(recommended_items[:k]):
                if item in target_set:
                    dcg += 1.0 / np.log2(i + 2)
                    found_targets += 1

                    # 如果已经找到所有目标物品，提前终止
                    if found_targets == len(target_items):
                        break

            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_results[k].append(ndcg)

    # 计算平均值
    final_results = {}
    for k in topk_list:
        final_results[f'T-HR@{k}'] = np.mean(hr_results[k])
        final_results[f'T-NDCG@{k}'] = np.mean(ndcg_results[k])

    return final_results

def get_target_users_and_unrated_items(num_users, num_items, trainMatrix, target_items):
    """
    寻找普通用户中与目标物品没有交集的用户，并返回这些用户及其未交互物品

    Returns:
        target_users: 列表，与目标物品没有交集的普通用户ID
        unrated_items_dict: 字典，键为用户ID，值为该用户未交互的物品列表
    """
    target_users = []
    unrated_items_dict = {}


    # 遍历所有原始普通用户（不包括后来添加的假用户）
    for user_id in range(num_users):
        # 获取该用户交互过的物品
        user_interactions = trainMatrix[user_id]
        rated_items = set(user_interactions.nonzero()[1])  # 获取用户交互过的物品索引

        # 检查用户是否与任何目标物品有交互
        has_target_interaction = any(item in rated_items for item in target_items)

        # 如果用户没有与任何目标物品交互，则将其加入目标用户列表
        if not has_target_interaction:
            target_users.append(user_id)

            # 获取该用户的所有未交互物品
            all_items = set(range(num_items))
            unrated_items = list(all_items - rated_items)
            unrated_items_dict[user_id] = unrated_items

    return target_users, unrated_items_dict
