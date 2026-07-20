
import sys
sys.path.append('./AT_AWP_CWB0826')  
from Dataset import Dataset
import utils
import argparse
import utils_LightGCN
import numpy as np

from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import torch
import load_dataset
import time
from copy import deepcopy
import os
from LightGCN import LightGCN
import gc  # 引入垃圾回收
import random
import sys
sys.path.append('./AT_AWP_CWB0826')

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

torch.cuda.set_device(device)

def seed_everything(seed=26):
    """
    固定所有可能的随机种子，保证实验可复现
    """
    # 1. Python random
    random.seed(seed)
    
    # 2. OS environment (Hash seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. NumPy
    np.random.seed(seed)
    
    # 4. PyTorch CPU
    torch.manual_seed(seed)
    
    # 5. PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 如果使用多卡
        
        # 6. CuDNN 确定性算法 (关键！因为你用了 LightGCN)
        # 这会让卷积变慢一点点，但能保证结果一致
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Run LightGCN_TWP.")
    parser.add_argument("--enable_lat", nargs="?", default=True)
    # parser.add_argument("--epsilon", nargs="?", default=0.5)
    parser.add_argument("--alpha", nargs="?", default=1)
    parser.add_argument("--pro_num", nargs="?", default=25,
                        choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    parser.add_argument("--layerlist", nargs="?", default="all")
    parser.add_argument("--adv", nargs="?", default=True)
    parser.add_argument("--adv_reg", nargs="?", default=1)
    parser.add_argument("--adv_type", nargs="?", default="fgsm",
                        choices=['fgsm', 'bim', 'pgd', 'mim'])
    parser.add_argument("--norm", nargs="?", default="linf",
                        choices=['linf', 'l2'])
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # lastfm yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument("--fake_users_file", nargs="?", default="Data/ml1m_attack/ml1m_2%/ml1m.random_head_0.02.json",
                        help="Input data path.")
    parser.add_argument("--attack_type", nargs="?", default="random",
                        choices=['entars', 'wmf'])
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--fname', default='results/LightGCN/TWP', type=str)
    # 25 控制有目标噪声大小的参数##################
    parser.add_argument("--alpha_awp", nargs="?", default=0)
    # 0.001,0.005##############################要调，控制加在层上的噪声大小
    parser.add_argument('--awp-gamma', default=0.002, type=float)
    # 是否对目标 item 做攻击（尾项攻击、push/ nuke 攻击相关）
    parser.add_argument('--para_target_attack', default=True, type=bool)
    parser.add_argument('--lr', default=0.5, type=float)
    parser.add_argument('--epsilon', default=0.01, type=float)######################控制扰动大小
    parser.add_argument('--cnn_start_epoch', type=int, default=99, metavar='N', help='number of epochs to pretrain')
    parser.add_argument("--num_to_select", type=int, default=3,help="num_to_select")
    # === 新增 seed 参数 ===
    parser.add_argument("--seed", type=int, default=26, help="Random seed for reproducibility") 
    return parser.parse_args()

def train(model, proxy_adv, path, train_dataset, testRatings, testNegatives, target_items, target_users, unrated_items_dict):
    lr = args.lr
    epoches = args.epochs
    batch_size = 1000
    best_hr10, best_ndcg10, bestepoch = 0, 0, -1

    model.train()
    print("oral_lr=0.5,有学习策略,gamma0.001")
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    proxy_opt = optim.Adagrad(proxy_adv.parameters(),
                              lr=0.01, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    cnn_start_epoch = args.cnn_start_epoch
    # 存储当前每层噪声大小
    temp_awp_gamma = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    train_loader = Data.DataLoader(
            train_dataset.train_group, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    for epoch in range(epoches):

        model.train()
       # === 逻辑判断 ===
        if epoch < cnn_start_epoch:
            # 阶段 1: MF 预训练
            is_pretrain_stage = True
            current_stage_name = "MF Pretrain"
        else:
            # 阶段 3: APR 对抗训练
            is_pretrain_stage = False
            current_stage_name = "CNN"
        
        # 可以在 log 里打印一下当前状态，确认是否生效
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Stage=[{current_stage_name}]")
        
        if epoch >= cnn_start_epoch:
            temp_awp_gamma = find_gamma(args, logger, model, train_dataset.train_group, train_loader, temp_awp_gamma)
            logger.info("args.awp_gamma_temp{},args.pro_num: {}".format(temp_awp_gamma, args.pro_num))
            awp_adversary = utils_LightGCN.AdvWeightPerturb_LightGCN_TWP(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=temp_awp_gamma)

        num_batch = 0
        for train_data in train_loader:
            train_data = train_data.to(device)
            user_ids = Variable(train_data[:, 0].to(device))
            pos_item_ids = Variable(train_data[:, 1].to(device))
            neg_item_ids = Variable(train_data[:, 2].to(device))

            optimizer.zero_grad()

            if epoch >= cnn_start_epoch:
                if args.para_target_attack and num_batch % 20 == 0:
                    # 随机选取5个目标项目   
                    # 使用所有项目范围（0到nItems-1）进行随机选择
                    ran_target_items = random.sample(
                        range(args.nItems), min(5, args.nItems))
                    # 确保target_items是列表格式
                    ran_target_items = list(ran_target_items)
                    # logger.info(f"Randomly selected target items for attack: {ran_target_items}")
                    awp = awp_adversary.calc_awp(args, user_ids=user_ids, pos_item_ids=pos_item_ids, neg_item_ids=neg_item_ids,
                                                 target_items=ran_target_items, alpha_awp=args.alpha_awp)
                    awp_adversary.perturb(awp)
                # train convncf
            loss = model(user_ids, pos_item_ids, neg_item_ids, is_pretrain_stage)


            num_batch += 1
            loss.backward()
            optimizer.step()

            if epoch >= cnn_start_epoch:
                awp_adversary.restore(awp)

            # 及时释放不再使用的张量
            torch.cuda.empty_cache()

        scheduler.step()

        if epoch == 5 or epoch >= cnn_start_epoch:
        # if epoch >=2:
            t1 = time.time()

            hits10, ndcgs10, maps10, mrrs10 = utils_LightGCN.evaluate_model_TWP(
                model, testRatings, testNegatives, 10, 20, 50, 100, 1, device)
            hr10, ndcg10, map10, mrr10 = np.array(hits10).mean(), np.array(
                ndcgs10).mean(), np.array(maps10).mean(), np.array(mrrs10).mean()

            logger.info(f"Epoch {epoch}: HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, mrrs10={mrr10:.8f} [{time.time()-t1:.1f}s]")



            if hr10 >= best_hr10:
                best_hr10 = hr10
                bestepoch = epoch
                best_ndcg10 = ndcg10

                results = utils_LightGCN.calculate_target_metrics_TWP(model, dataset.target_items, dataset.target_users,
                                            dataset.unrated_items_dict, topk_list=[10, 20, 50, 100])
                for metric, value in results.items():
                    if metric == f"T-HR@50":
                        target_hr = value   
                    if metric == "T-NDCG@50":
                        target_ndcg = value
                    logger.info(f"{metric}: {value:.8f}")
                logger.info(f"Training {epoch}_ Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")
                
                os.makedirs(os.path.join(path, 'pretrained'), exist_ok=True)
                best_model_path = os.path.join(path, 'pretrained', f"{args.dataset}_LightGCN_TWP_{time.time()}.pth")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best checkpoint to: {best_model_path} (HR10={best_hr10:.8f})")
                logger.info("------------------------------------------------------------------------------")
        torch.cuda.empty_cache()
    logger.info(
        f"Best epoch {bestepoch+1}: HR10={best_hr10:.8f}, NDCG10={best_ndcg10:.8f}")

    # Optional: final load best checkpoint and print final target HR
    if best_model_path and os.path.exists(best_model_path):
        logger.info("=" * 60)
        logger.info("Loading best checkpoint for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        results = utils_convNCF.calculate_target_metrics_TWP(
            model,
            target_items,
            target_users,
            unrated_items_dict,
            topk_list=[10, 20, 50, 100],
            device=device,
        )

        hits10, ndcgs10, maps10, mrrs10 = utils_convNCF.evaluate_model_TWP(
            model,
            testRatings,
            testNegatives,
            topK=10,
            topK1=20,
            topK2=50,
            topK3=100,
            device=device,
        )
        hr10 = float(np.array(hits10).mean())
        ndcg10 = float(np.array(ndcgs10).mean())
        map10 = float(np.array(maps10).mean())
        mrr10 = float(np.array(mrrs10).mean())
        logger.info(
            f"Final Test: HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, MAP10={map10:.8f}, MRR10={mrr10:.8f}  [{time.time()-t1:.1f}s]")

        for metric, value in results.items():
            if metric == f"T-HR@50":
                target_hr = value   
            if metric == "T-NDCG@50":
                target_ndcg = value
            logger.info(f"Final {metric}: {value:.8f}")
        logger.info(f"FinalAAAAAAAA Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")
        logger.info("------------------------------------------------------------------------------")


def find_gamma(args, logger, model_adv, train_group, train_loader, temp_awp_gamma):

    logger.info(args.dataset)

    layer_sharpness_dict = layer_sharpness(
        args, model_adv, train_group, train_loader)
    layer_sharpness_dict111 = list(layer_sharpness_dict.items())
    tensors = [torch.tensor(item[1], device=device)
               for item in layer_sharpness_dict111]
    tensor_array = torch.stack(tensors)

    # 定义基础的 gamma_max 和调整系数
    base_gamma_max = args.awp_gamma

    for i in range(9):
        T = random.uniform(0.8, 1.2)
        # 检查数值稳定性
        if torch.isnan(tensor_array[i]) or torch.isinf(tensor_array[i]):
            temp_awp_gamma[i] = base_gamma_max * 0.1
            continue

        # 计算脆弱程度的统计信息
        min_sharpness = torch.min(tensor_array)
        max_sharpness = torch.max(tensor_array)

        # 避免除零
        if max_sharpness <= min_sharpness:
            temp_awp_gamma[i] = base_gamma_max * 0.1
            continue

        # 基于排序的分配策略
        # 1. 对脆弱程度进行排序
        sorted_indices = torch.argsort(tensor_array)
        sorted_sharpness = tensor_array[sorted_indices]

        # 2. 找到当前层的排名
        current_rank = torch.where(sorted_indices == i)[0][0].item()

        # 3. 根据排名分配权重（排名越高，脆弱程度越高，噪声越大）
        # 使用指数函数增强高排名的权重
        rank_weight = ((current_rank + 1) / len(tensor_array)) ** 2  # 平方增强差异

        # 4. 添加随机扰动
        noise_ratio = rank_weight * T

        # 5. 将噪声大小映射到[0, base_gamma_max]范围
        temp_awp_gamma[i] = noise_ratio * base_gamma_max

        # 确保在合理范围内
        temp_awp_gamma[i] = max(temp_awp_gamma[i], base_gamma_max * 0.05)
        temp_awp_gamma[i] = min(temp_awp_gamma[i], base_gamma_max)

        # 可选：记录分配结果用于调试
        logger.debug(f"Layer {i}: sharpness={tensor_array[i].item():.6f}, "
                     f"rank={current_rank+1}/{len(tensor_array)}, "
                     f"noise_ratio={noise_ratio:.3f}, "
                     f"gamma={temp_awp_gamma[i]:.6f}")

    for i in range(9):
        # 按照比例计算 temp_awp_gamma[i] 的值
        if temp_awp_gamma[i] == 0:
            temp_awp_gamma[i] = 1e-6

    # 设置要随机选择的元素数量
    num_to_select = args.num_to_select  # 例如，选择2个元素

    # 将temp_awp_gamma转换为PyTorch张量
    gamma_tensor = torch.tensor(temp_awp_gamma, device=device)
    # 确保第一层和第二层有相同的gamma值（取两者的平均值）
    first_layer_gamma = (temp_awp_gamma[0] + temp_awp_gamma[1]) / 2
    temp_awp_gamma[0] = first_layer_gamma
    temp_awp_gamma[1] = first_layer_gamma

    # 计算归一化的概率（softmax）
    probabilities = torch.softmax(gamma_tensor, dim=0)
    # 按照概率分布随机选择num_to_select个索引（不重复）
    selected_indices = torch.multinomial(
        probabilities, num_to_select, replacement=False).tolist()

    # 检查是否选中了第一层或第二层
    selected_layer_0 = 0 in selected_indices
    selected_layer_1 = 1 in selected_indices

    # 如果只选中了其中一个，自动添加另一个，并移除一个其他层（概率最小的）
    if selected_layer_0 and not selected_layer_1:
        # 添加第二层
        selected_indices.append(1)
        # 从选中的索引中找出非0和1的索引中概率最小的那个
        other_indices = [idx for idx in selected_indices if idx not in [0, 1]]
        if other_indices:
            # 找到这些其他索引中概率最小的
            min_prob_index = min(other_indices, key=lambda x: probabilities[x])
            selected_indices.remove(min_prob_index)
    elif selected_layer_1 and not selected_layer_0:
        # 添加第一层
        selected_indices.append(0)
        other_indices = [idx for idx in selected_indices if idx not in [0, 1]]
        if other_indices:
            min_prob_index = min(other_indices, key=lambda x: probabilities[x])
            selected_indices.remove(min_prob_index)

    # 确保不超过选择数量限制
    if len(selected_indices) > num_to_select:
        # 如果因为添加而超过了限制，随机移除一个非0和1的索引
        other_indices = [idx for idx in selected_indices if idx not in [0, 1]]
        if other_indices:
            min_prob_index = min(other_indices, key=lambda x: probabilities[x])
            selected_indices.remove(min_prob_index)

    # 创建新的gamma数组，只保留随机选择的2个对应的值，其他设为0
    new_temp_awp_gamma = [0.0] * 9
    for idx in selected_indices:
        new_temp_awp_gamma[idx] = temp_awp_gamma[idx]

    # 记录选择结果用于调试
    logger.info(
        f"Randomly selected {num_to_select} layers: {selected_indices}")
    logger.info(f"Original gamma values: {temp_awp_gamma}")
    logger.info(f"Selected gamma values: {new_temp_awp_gamma}")
    logger.info(f"Selection probabilities: {probabilities.tolist()}")

    return new_temp_awp_gamma


def layer_sharpness(args, model_adv, train_group, train_loader):
    # 选取一组随机目标项目进行攻击
    t1 = time.time()
    if args.dataset == "ml-1m":
        num_tcal = 5000
    if args.dataset == "lastfm" or args.dataset == "AMusic":
        num_tcal = 330   #
    if args.dataset == "yelp" or args.dataset == "gowalla":
        num_tcal = 3300

    epsilon = args.epsilon
    layer_sharpness_dict = {}
    SAFE_BATCH_SIZE = 256 

    def apply_norm_scaling(diff, init_param, eps):
        # flatten
        d = diff.view(diff.size(0), -1)
        p = init_param.view(init_param.size(0), -1)

        d_norm = torch.norm(d, p=2, dim=1)            # ||diff||₂
        p_norm = torch.norm(p, p=2, dim=1)            # ||param||₂

        scale = eps * (p_norm / (d_norm + 1e-12))     # 避免除 0

        # 任何 NaN / Inf → 强制设为 0（不扰动该子行）
        scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
        scale_reshaped = scale.view(-1, *([1] * (diff.dim() - 1)))

        return diff * scale_reshaped
     # 仅统计含权重的模块，比如一些 ReLU、Dropout、BatchNorm 之类的层
    # 它们也在模型里，但它们本身没有可训练权重，也就不需要计算“脆弱度”
    for name, module in model_adv.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
            '''
            layer_sharpness_dict = {
                "conv1": -1e10,
                "conv2": -1e10,
                "fc1": -1e10
            }
            '''
            clean = name.replace("module.", "")  # 修改点3
            # layer_sharpness_dict[name] = -1e10
            layer_sharpness_dict[clean] = -1e10

    # ---------------- 统一参数命名（移除 DataParallel 的 module. 前缀） ----------------
    # 找有权重的
    param_map = {}
    for name, param in model_adv.named_parameters():
        # print(name, param.shape)
        # clean = name.split("module.")[-1]  # 删除前缀 module.
        clean = name.replace("module.", "")  # 修改点1
        if clean.endswith(".weight"):      # 仅使用 weight
            '''
            param_map = {
                "conv1.weight": tensor(...),
                "conv2.weight": tensor(...),
                "fc1.weight": tensor(...)
            }
            '''
            param_map[clean] = param
            # print("=====================================================")
            # print(name, param.shape)

    # 过滤 layer_sharpness_dict 只保留真正存在的权重层
    valid_layers = []
    for layer in layer_sharpness_dict.keys():
        target_name = f"{layer}.weight"
        # clean_layer = layer.replace("module.", "") # 修改点2
        # target_name = f"{clean_layer}.weight" # 修改点2
        if target_name in param_map:
            valid_layers.append(target_name)  # 即是脆弱层，又有权重

    if len(valid_layers) == 0:
        args.logger.info("未检测到可用权重层，退出 layer_sharpness 计算")
        return {}

    # 无目标噪声所以是随机选择5个target items?xq1112
    # 3. 随机选 5 个“目标物品”，后面攻击就看模型能不能把它们预测得更靠前
    num_target_items = 5  # 选择50个随机目标项目
    target_items = np.random.choice(args.nItems, size=min(
        num_target_items, args.nItems), replace=False)

    target_users, unrated_items_dict = utils_LightGCN.get_target_users_and_unrated_items(
        args.nUsers, args.nItems, train_group, target_items)
    # 计算初始指标 (确保无梯度)
    with torch.no_grad():
        results = utils_LightGCN.calculate_target_metrics_TWP(
            model_adv, target_items, target_users, unrated_items_dict,
            topk_list=[10, 20, 50, 100], device=device)
    for metric, value in results.items():
        if metric == f"T-HR@50":
            target_hr = value
        if metric == "T-NDCG@50":
            target_ndcg = value
        args.logger.info(f"{metric}: {value:.8f}")
    args.logger.info(
        f"Training_init_ Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")

    original_target_hr = target_hr

    process_model = deepcopy(model_adv).to(device)
    clean_state_dict = model_adv.state_dict()

    # 开始“逐层”做对抗扰动
    # 第一层和第二层应该一起加噪声
   # 获取前两个层的名称
    # 是否启用“前两层共同噪声”机制（你要求关闭，因此写为 False）
    bind_first_two = True
    top2_layers = []
    
    # ================= 阶段 1: 前两层共同扰动 =================
    if bind_first_two and len(valid_layers) >= 2:
        top2_layers = valid_layers[:2]
        other_layers = valid_layers[2:]
        args.logger.info(f"绑定噪声层: {top2_layers}")

        # 1. 重置模型状态
        process_model.load_state_dict(clean_state_dict)
        
        # 2. 映射参数并保存初始值
        cloned_param_map = {
            name.split("module.")[-1]: p
            for name, p in process_model.named_parameters()
            if name.split("module.")[-1].endswith(".weight")
        }
        
        init_params = {}
        valid_bind = True
        for ln in top2_layers:
            if ln in cloned_param_map:
                init_params[ln] = cloned_param_map[ln].detach().clone()
            else:
                valid_bind = False
                break
        
        if valid_bind:
            # 3. 设置梯度与优化器 (关键优化点：只传需要训练的参数给优化器)
            trainable_params = []
            for name, p in cloned_param_map.items():
                if name in top2_layers:
                    p.requires_grad = True
                    trainable_params.append(p)
                else:
                    p.requires_grad = False
            
            # 这里的 trainable_params 列表很小，Adagrad 状态占用极小
            optimizer = torch.optim.Adagrad(trainable_params, lr=args.lr, weight_decay=1e-6)

            max_target_hr_improvement = -1000.0
            
            # 4. 训练循环
            process_model.train()
            cal_num = 0
            
            # 为了防止 train_loader 迭代器产生的内存驻留，尽量简化循环
            train_iter = iter(train_loader)
            while cal_num < num_tcal:
                try:
                    train_data = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    train_data = next(train_iter)

                # 获取当前这一批次的所有用户
                all_user_ids = train_data[:, 0]
                    
                # --- 新增：内部切片循环 (Mini-Batch Slicing) ---
                total_samples = len(all_user_ids)
                    
                # 按照 SAFE_BATCH_SIZE 进行切分
                for start_idx in range(0, total_samples, SAFE_BATCH_SIZE):
                    end_idx = min(start_idx + SAFE_BATCH_SIZE, total_samples)
                        
                    # 1. 取出一小部分用户
                    current_batch_users = all_user_ids[start_idx:end_idx].to(device)
                        
                    optimizer.zero_grad() # 记得清零

                    # 2. 构造数据 (逻辑和之前一样，但只针对 current_batch_users)
                    unique_users = torch.unique(current_batch_users).detach()
                        
                    # 这里的 target_items_tensor 只有 5 个，不大
                    target_items_tensor = torch.tensor(target_items, device=device)
                        
                    num_users = len(unique_users)
                    num_targets = len(target_items_tensor)
                        
                    # 扩展： [Batch_Size] -> [Batch_Size * 5]
                    user_seq = unique_users.repeat_interleave(num_targets)
                    pos_item_seq = target_items_tensor.repeat(num_users)

                    # 负采样 (逻辑不变)
                    all_items = torch.arange(args.nItems, device=device)
                    mask = torch.ones_like(all_items, dtype=torch.bool)
                    mask[target_items_tensor] = False
                    candidate_items = all_items[mask]
                        
                    rand_idx = torch.randint(
                        low=0, 
                        high=candidate_items.size(0), 
                        size=pos_item_seq.shape, 
                        device=device
                    )
                    neg_item_seq = candidate_items[rand_idx]

                    # 3. 前向传播 (现在的 input size 是安全的)
                    # 这里的 loss 是这一小批的平均 loss
                    loss = process_model(user_seq, pos_item_seq, neg_item_seq, False)
                        
                    # 4. 反向传播
                    loss.backward()
                        
                    # 5. 更新参数
                    # 注意：如果原本是想积累整个大 Batch 的梯度再 update，这里会有区别。
                    # 但在对抗攻击场景下，频繁 update 通常效果更好或没区别。
                    optimizer.step()

                    # 外层计数增加 (原本逻辑是处理完一个 train_loader 的 batch 算一次)
                cal_num += 1
                if cal_num >= num_tcal:
                    break
            
            # 5. 应用扰动
            process_model.eval()
            # 获取最新的 state_dict (注意：不要深拷贝整个 dict，操作引用即可)
            sd = process_model.state_dict()
            
            # 临时字典用于计算 diff，避免修改 DataParallel 的 key 导致混乱
            # 这里的 sd 已经是引用，直接修改 sd 里的 tensor 会生效
            # 但我们需要先提取出干净的 key 对应关系
            
            with torch.no_grad(): # 扰动计算不需要梯度
                for ln in top2_layers:
                    # 找到对应的 param (处理 module. 前缀)
                    # 遍历查找比较慢，但因为层数少没关系，或者建立映射
                    # 简单起见，直接用之前存的 cloned_param_map 的 value (它就是 parameter tensor)
                    # 注意：optimizer.step() 已经修改了 parameter 的值
                    
                    curr_param = cloned_param_map[ln] # 这是引用
                    diff = curr_param - init_params[ln]
                    diff = apply_norm_scaling(diff, init_params[ln], epsilon)
                    curr_param.copy_(init_params[ln] + diff) # 使用 copy_ 原地修改

            # 6. 评估
            with torch.no_grad(): # 必须加 no_grad
                results = utils_convNCF.calculate_target_metrics_TWP(
                    process_model, target_items, target_users, unrated_items_dict,
                    topk_list=[10, 20, 50, 100], device=device)

            attacked_target_hr = results.get("T-HR@50", 0.0)
            hr_improvement = attacked_target_hr - original_target_hr
            max_target_hr_improvement = hr_improvement
            
            for ln in top2_layers:
                layer_sharpness_dict[ln.replace(".weight", "")] = max_target_hr_improvement * 100
                args.logger.info(f"[Top-2 layer] {ln.replace('.weight',''):30} HR Imp: {hr_improvement:.6f}")

            # 7. 清理
            del optimizer
            del init_params
            del trainable_params
            torch.cuda.empty_cache()
            
    else:
        top2_layers = []
        other_layers = valid_layers[:]
        args.logger.info("逐层处理模式")

    # ================= 阶段 2: 逐层扰动 =================
    for layer_name in other_layers:
        layer_name_clean = layer_name
        
        # 1. 重置模型回到干净状态
        process_model.load_state_dict(clean_state_dict)
        
        # 2. 重新构建 param map (因为 process_model 的某些引用可能在上面被操作过，安全起见重置指向)
        cloned_param_map = {}
        for name, p in process_model.named_parameters():
            clean = name.split("module.")[-1]
            if clean.endswith(".weight"):
                cloned_param_map[clean] = p

        if layer_name_clean not in cloned_param_map:
            continue

        # 3. 锁定参数，只优化当前层
        init_param = cloned_param_map[layer_name_clean].detach().clone()
        
        trainable_params = []
        for name, p in cloned_param_map.items():
            if name == layer_name_clean:
                p.requires_grad = True
                trainable_params.append(p)
            else:
                p.requires_grad = False

        # 优化点：只传一个参数进优化器
        optimizer = optim.Adagrad(trainable_params, lr=args.lr, weight_decay=1e-2)
        
        # 4. 训练
        process_model.train()
        cal_num = 0
        train_iter = iter(train_loader)
        
        while cal_num < num_tcal:
            try:
                train_data = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                train_data = next(train_iter)

            # 获取当前这一批次的所有用户
            all_user_ids = train_data[:, 0]
                
            # --- 新增：内部切片循环 (Mini-Batch Slicing) ---
            total_samples = len(all_user_ids)
                
            # 按照 SAFE_BATCH_SIZE 进行切分
            for start_idx in range(0, total_samples, SAFE_BATCH_SIZE):
                end_idx = min(start_idx + SAFE_BATCH_SIZE, total_samples)
                    
                # 1. 取出一小部分用户
                current_batch_users = all_user_ids[start_idx:end_idx].to(device)
                    
                optimizer.zero_grad() # 记得清零

                # 2. 构造数据 (逻辑和之前一样，但只针对 current_batch_users)
                unique_users = torch.unique(current_batch_users).detach()
                    
                # 这里的 target_items_tensor 只有 5 个，不大
                target_items_tensor = torch.tensor(target_items, device=device)
                    
                num_users = len(unique_users)
                num_targets = len(target_items_tensor)
                    
                # 扩展： [Batch_Size] -> [Batch_Size * 5]
                user_seq = unique_users.repeat_interleave(num_targets)
                pos_item_seq = target_items_tensor.repeat(num_users)

                # 负采样 (逻辑不变)
                all_items = torch.arange(args.nItems, device=device)
                mask = torch.ones_like(all_items, dtype=torch.bool)
                mask[target_items_tensor] = False
                candidate_items = all_items[mask]
                    
                rand_idx = torch.randint(
                    low=0, 
                    high=candidate_items.size(0), 
                    size=pos_item_seq.shape, 
                    device=device
                )
                neg_item_seq = candidate_items[rand_idx]

                # 3. 前向传播 (现在的 input size 是安全的)
                # 这里的 loss 是这一小批的平均 loss
                loss = process_model(user_seq, pos_item_seq, neg_item_seq, False)
                    
                # 4. 反向传播
                loss.backward()
                    
                # 5. 更新参数
                # 注意：如果原本是想积累整个大 Batch 的梯度再 update，这里会有区别。
                # 但在对抗攻击场景下，频繁 update 通常效果更好或没区别。
                optimizer.step()

                # 外层计数增加 (原本逻辑是处理完一个 train_loader 的 batch 算一次)
            cal_num += 1
            if cal_num >= num_tcal:
                break

        # 5. 应用扰动
        process_model.eval()
        with torch.no_grad():
            curr_param = cloned_param_map[layer_name_clean]
            diff = curr_param - init_param
            diff = apply_norm_scaling(diff, init_param, epsilon)
            curr_param.copy_(init_param + diff)

        # 6. 评估
        with torch.no_grad():
            results = utils_LightGCN.calculate_target_metrics_TWP(
                process_model, target_items, target_users, unrated_items_dict,
                topk_list=[10, 20, 50, 100], device=device,
            )
        
        attacked_target_hr = results.get("T-HR@50", 0.0)
        hr_improvement = attacked_target_hr - original_target_hr
        
        layer_sharpness_dict[layer_name_clean.replace(".weight", "")] = hr_improvement * 100
        args.logger.info(f"{layer_name_clean.replace('.weight',''):30} HR Imp: {hr_improvement*100:.6f}")

        # 7. 显式清理显存
        del optimizer
        del trainable_params
        del init_param
        # 强制 Python 进行垃圾回收，并清理 PyTorch 缓存
        gc.collect()
        torch.cuda.empty_cache()

    # 最后删除复用的模型
    del process_model
    torch.cuda.empty_cache()

    args.logger.info(layer_sharpness_dict)
    return layer_sharpness_dict


if __name__ == '__main__':

    # torch.set_num_threads(12)
    args = parse_args()
    seed_everything(args.seed)
    # Directory and logger
    path = os.path.join(args.fname, args.dataset, args.attack_type)
    os.makedirs(path, exist_ok=True)
    logger = utils.create_logger(os.path.join(path, 'output.log'))
    args.logger = logger
    args.device = device

    logger.info('Data Train_loading...')
    train_dataset = load_dataset.Load(
        args.data_path + args.dataset, args.fake_users_file, args.attack_type)
    logger.info('Data loaded')
    dataset = Dataset(args.data_path + args.dataset,
                      fake_users_file=args.fake_users_file)
    _, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    logger.info('=' * 50)

    logger.info('Model initializing...')
    args.nUsers = int(max(train_dataset.train_group[:, 0])) + 1
    args.nItems = int(max(train_dataset.train_group[:, 1])) + 1
    model = LightGCN(args.nUsers, args.nItems, device)
    proxy_adv = LightGCN(args.nUsers, args.nItems, device)
    # awp_adversary = AdvWeightPerturb1(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=awp_gamma)
    logger.info('Model initialized')

    logger.info('=' * 50)

    logger.info('Model training...')
    # train(awp_gamma)
    train(model, proxy_adv, path, train_dataset, testRatings, testNegatives, target_items=dataset.target_items,
          target_users=dataset.target_users, unrated_items_dict=dataset.unrated_items_dict)
    logger.info('Model trained')
