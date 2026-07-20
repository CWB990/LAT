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
import utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import pandas as pd
from scipy import sparse
import numpy as np


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
        
    hits10, ndcgs10,maps10, mrrs10, hits20, ndcgs20,maps20, mrrs20, hits50, ndcgs50, maps50, mrrs50 = [], [], [], [], [], [], [], [],[], [], [], []
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
    label[-1] = 1
    # ---
    dst = TestDataset_adv(users, items, label)
    ldr = torch.utils.data.DataLoader(dst, batch_size=256, shuffle=False)

    _model.eval()
    predictions = [None] * len(dst)
    total = 0
    with torch.no_grad():
        for ui, ii, lbl in ldr:
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)
            bsz = ui.size(0)
            _,_,_,_,_,_,_, ri = _model(ui, ii, lbl)

            ri = ri.squeeze().cpu().tolist()
            predictions[total:total+bsz] = ri

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

# def cifar10(root):
#     train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
#     test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
#     return {
#         'train': {'data': train_set.data, 'labels': train_set.targets},
#         'test': {'data': test_set.data, 'labels': test_set.targets}
#     }

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




def evalulate_robustness(args, model, ldr, opt, epsilon, testRatings, testNegatives, topK, topK1, topK2, topK3):
    emb_backup = {}
    criterion = torch.nn.BCELoss()
    is_first_attack = True
    start_time = time.time()
    cloned_model = deepcopy(model)
    rounds = 10
    for name, param in cloned_model.named_parameters():
            param.requires_grad = True
            
    optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

    args.attack = 'fgsm' #################################
    cloned_model.train()
    if args.attack == 'fgsm':
        for ui, ii, lbl in ldr:
            optimizer.zero_grad()
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            loss1,_,_,_,_,_,_,outputs= cloned_model(ui, ii,lbl)
            outputs = outputs.squeeze()
            loss =  criterion(outputs, lbl)
            loss.backward()
            
            with torch.no_grad():
                for name, param in cloned_model.named_parameters():
                    if is_first_attack:
                        emb_backup[name] = param.data.clone()
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = -param.grad / norm
                        param.data.add_(r_at)
                        r = param.data - emb_backup[name]
                        if torch.norm(r) > epsilon:
                            r = epsilon * r / torch.norm(r)
                        param.data = emb_backup[name] + r
                is_first_attack = False
    # print(time.time() - start_time)
    
    if args.attack == 'pgd':
        for epoch in range(rounds):
                start_time = time.time()
                if is_first_attack:
                    # 备份原始模型参数
                    for name, param in cloned_model.named_parameters():
                        emb_backup[name] = param.data.clone()
                        if cloned_model and param.requires_grad:
                            # 初始化随机扰动
                            random_noise = epsilon * torch.randn_like(param)
                            param.data.add_(random_noise)
                            # 确保扰动不超过epsilon
                            param_diff = param.data - emb_backup[name]
                            if torch.norm(param_diff) > epsilon:
                                param.data = emb_backup[name] + epsilon * param_diff / torch.norm(param_diff)
                
                is_first_attack = False

                for ui, ii, lbl in ldr:
                    ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                    cloned_model.train()
                    loss,_,_,_,_,_,_,output= cloned_model(ui, ii,lbl)
                    yc = output.squeeze()
                    loss = -criterion(yc, lbl)
                    opt.zero_grad()
                    loss.backward()
                    # 对模型参数应用扰动
                    with torch.no_grad():
                        for name, param in cloned_model.named_parameters():
                            if param.requires_grad:
                                # 计算扰动
                                norm = torch.norm(param.grad)
                                if norm != 0 and not torch.isnan(norm):
                                    # 添加扰动
                                    delta = epsilon/rounds * param.grad / norm
                                    param.data.add_(delta)
                                    # 确保扰动在epsilon范围内
                                    param_diff = param.data - emb_backup[name]
                                    if torch.norm(param_diff) > epsilon:
                                        param.data = emb_backup[name] + epsilon * param_diff / torch.norm(param_diff)
            
            # print(time.time()-start_time)
    
    
    cloned_model.eval()
    # 检查模型性能
    t1 = time.time()
    hits10, ndcgs10, maps10, mrrs10 = evaluate_model(cloned_model, testRatings, testNegatives, topK, topK1, topK2, topK3, num_thread=1, device=device)

    # # 恢复模型的原始参数
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         if name in emb_backup:
    #             param.data = emb_backup[name]
    del cloned_model
    hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
       
    # print(f"Epoch : HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f}, HR20={hr20:.4f}, NDCG20={ndcg20:.4f}, mrrs20={mrr20:.4f}, HR50={hr50:.4f}, NDCG50={ndcg50:.4f}, mrrs50={mrr50:.4f} [{time.time()-t1:.1f}s]")
    return hr10





def interpolation(args, logger, init_sd, ft_sd, model, trainloader, criterion,save_dir):
    alphas = np.arange(0, 1.1, 0.1)

    # alphas = np.arange(0, 0.21, 0.01)
    
    records = []
    best_performance = 0
    best_checkpoint_path = None  # 在使用之前给变量一个初始值
    integrated_state_dict = {}
    for alpha in alphas:
        model_dict = {}
        for name, _ in init_sd.items():
            model_dict[name] = alpha * ft_sd[name] + (1 - alpha) * init_sd[name]

        # 临时保存模型参数
        temp_checkpoint_path = save_dir + "finetune_{:.3f}_params.pth".format(alpha)
        torch.save(model_dict, temp_checkpoint_path)
        # torch.save(model_dict, save_dir + "finetune_{:.3f}_params.pth".format(alpha))

        model.load_state_dict(model_dict)
        hits10, ndcgs10, _, _ = evaluate_model(model, args.testRatings, args.testNegatives, 
                                                                         args.topK,args.topK1,args.topK2,args.topK3, num_thread=1, device=device)
        hr10 = np.array(hits10).mean()
        test_robust_acc = utils.evalulate_robustness(args, model,trainloader,args.optimizer,args.epsilon,
                                                   args.testRatings, args.testNegatives, args.topK,args.topK1,args.topK2,args.topK3)

        logger.info("==> Alpha: {:.4f}, hits10: {:.4f}%, test robust hit10: {:.4f}%".format(alpha, hr10*100, test_robust_acc*100))
        records.append((hr10, test_robust_acc))

        # 使用hr10和test_robust_acc的综合值来评估性能，这里简单地取它们的平均值作为示例
        current_performance = (hr10 + test_robust_acc) / 2
        if alpha == 0:
            test_robust_acc_init = test_robust_acc
        # 更新最优模型


        if test_robust_acc >=test_robust_acc_init:
            if current_performance > best_performance:
                best_performance = current_performance
                best_alpha = alpha
                if best_checkpoint_path is not None and os.path.isfile(best_checkpoint_path):
                    try:
                        os.remove(best_checkpoint_path)
                    except Exception as e:
                        print(f"删除文件时出错：{e}")
                else:
                    print("未提供有效的 best_checkpoint_path,跳过删除步骤。")

                best_checkpoint_path = temp_checkpoint_path
                integrated_state_dict = model_dict
            # 注意：这里我们保留最优模型的checkpoint路径，以便后续使用，不再删除
            
    # 循环结束后，载入最优模型参数
    if best_checkpoint_path:
        best_model_dict = torch.load(best_checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_model_dict)
        logger.info("Loaded best model with alpha {:.4f}".format(best_alpha))


    return integrated_state_dict,records