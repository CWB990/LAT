#TWP的awp
import random
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20

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



def add_into_weights_con(model, diff,coeff,ratio=0.4): # 使用pgd更新模型参数
    names_in_diff = diff.keys()
    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff :
                param.add_(coeff[i] * diff[name])
                i = i + 1

def add_into_weights(model, diff, seeds,coeff=1.0,ratio=0.5): # 使用pgd更新模型参数
    names_in_diff = diff.keys()
    i = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff :
                if seeds[i] == 1:
                    param.add_(coeff * diff[name])


class AdvWeightPerturb_con(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb_con, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma


    def calc_awp(self, inputs_adv_u, inputs_adv_i, inputs_adv_lab, target_items, alpha_awp, temperature=0.1):
        # 确保alpha_awp是浮点数类型，避免类型错误
        if isinstance(alpha_awp, str):
            alpha_awp = float(alpha_awp)
        elif not isinstance(alpha_awp, (int, float)):
            alpha_awp = float(alpha_awp)
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        # 计算当前用户和target items的损失，然后拉近这个损失
        criterion = torch.nn.BCELoss()
        def binary_cross_entropy_individual(pred, target):
            # 使用数值稳定的实现
            pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
            return -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))   # 变成正数
        
        # unique_users = torch.tensor(torch.unique(inputs_adv_u.detach().clone()), device=inputs_adv_u.device).detach().clone()
        unique_users = torch.unique(inputs_adv_u).detach().clone()
        # 为每个用户构建与目标项目的序列
        target_items_tensor = torch.tensor(target_items, device=inputs_adv_i.device)
        # 计算需要处理的用户-项目对数量
        num_users = len(unique_users)
        num_targets = len(target_items_tensor)
        # total_pairs = num_users * num_targets

        # 批量构建用户-项目配对
        user_seq = unique_users.repeat_interleave(num_targets)  # 每个用户重复num_targets次
        item_seq = target_items_tensor.repeat(num_users)  # 目标项目重复num_users次
        label_seq = torch.ones_like(user_seq, dtype=torch.float)  # 目标标签为1
        

        
        # 通过模型计算用户-目标项目对的预测
        _,_,_,_,_,_,_,target_output = self.proxy(user_seq, item_seq, label_seq)
        target_yc = target_output.squeeze()
        # 计算有目标损失：希望模型对目标项目预测为1
        target_losses = criterion(target_yc, label_seq)
        
        _,_,_,_,_,_,_,output = self.proxy(inputs_adv_u, inputs_adv_i, inputs_adv_lab)
        yc = output.squeeze()
        
        # 计算每个样本的个体损失
        individual_losses = binary_cross_entropy_individual(yc, inputs_adv_lab)

        loss_ind = individual_losses.detach()
        
        # 获取唯一用户及其对应的索引和计数
        unique_users, inverse_indices, counts = torch.unique(
            inputs_adv_u, return_inverse=True, return_counts=True)
        
        # 计算每个用户的总损失
        user_loss_sum = torch.zeros_like(inputs_adv_u, dtype=loss_ind.dtype)
        user_loss_sum.scatter_add_(0, inverse_indices, loss_ind)
        
        # 计算用户平均损失
        mean_loss = user_loss_sum.mean()
        
        # 为每个样本获取对应的用户总损失
        user_total_loss = user_loss_sum[inverse_indices]
        eps = torch.zeros_like(inputs_adv_u, dtype=loss_ind.dtype)
        # 计算 eps
        num_user= 0
        for user in inputs_adv_u:
            eps[num_user] = mean_loss / (user_loss_sum[num_user] + 1e-9) - 1
            num_user = num_user + 1
        # eps = mean_loss / (user_total_loss + 1e-9) - 1
        
        loss_weights = torch.sigmoid(eps.to(inputs_adv_i.device))

        non_target_loss = (individual_losses * loss_weights).sum() * (1 - alpha_awp)  + target_losses * alpha_awp

        # 负号不能加太早，否则会导致损失的大小计算出错
        loss = -non_target_loss


        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()
        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)        
        return diff

      
    def perturb(self, diff):
        # 对序列中的每个元素乘以 1.0
        new_coeff = [1.0 * g for g in self.gamma]
        add_into_weights_con(self.model, diff,coeff=new_coeff)

    def restore(self, diff):
        # 对序列中的每个元素乘以 -1.0
        new_coeff = [-1.0 * g for g in self.gamma]
        add_into_weights_con(self.model, diff, coeff=new_coeff)


