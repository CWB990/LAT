#TWP最终使用版

import argparse
import random
import time
from bunch import Bunch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP  
from utils import *
from Dataset import Dataset

from evaluate import *
from utils_awp_TWP import AdvWeightPerturb_con


##计算每层的鲁棒性
def layer_sharpness(args, model_adv,train,testRatings,testNegatives, topK, topK1, topK2, topK3, trainloader):
    # 选取一组随机目标项目进行攻击
    t1 = time.time()
    epsilon = args.epsilon
    layer_sharpness_dict = {} 
    # 模型层遍历和锐度计算
    for name, module in model_adv.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if "sub" in name:
                continue
            layer_sharpness_dict[name] = -1e10  # 初始化为负值，用于记录最大命中率提升

    if args.dataset == "ml-1m":
        num_tcal = 5000
    if args.dataset == "lastfm" or args.dataset == "AMusic":
        num_tcal = 330   #
    if args.dataset == "yelp"or args.dataset == "gowalla":
        num_tcal = 3300

    ##无目标噪声所以是随机选择5个target items?xq1112
    # 3. 随机选 5 个“目标物品”，后面攻击就看模型能不能把它们预测得更靠前
    num_target_items = 5  # 选择50个随机目标项目
    nUsers, nItems = train.shape
    target_items = np.random.choice(nItems, size=min(num_target_items, nItems), replace=False)
    target_users, unrated_items_dict = get_target_users_and_unrated_items(nUsers, nItems, train, target_items)
    results = calculate_target_metrics(model_adv, target_items, target_users,
                                               unrated_items_dict, topk_list=[10, 20, 50, 100])
    for metric, value in results.items():
        if metric == f"T-HR@50":
                target_hr = value   
        if metric == "T-NDCG@50":
                target_ndcg = value
        args.logger.info(f"{metric}: {value:.8f}")
    args.logger.info(f"Training_init_ Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")
    
    original_target_hr = target_hr
    with torch.no_grad():
        model_adv.eval()
    

    #开始“逐层”做对抗扰动
    #########第一层和第二层应该一起加噪声
   # 获取前两个层的名称
    all_layer_names = [name for name, _ in model_adv.named_parameters() 
                    if "weight" in name and name[:-len(".weight")] in layer_sharpness_dict.keys()]
    top2_layers = all_layer_names[:2]  # 前两个层
    other_layers = all_layer_names[2:]  # 其他层

    args.logger.info(f"同时处理的层: {top2_layers}")
    args.logger.info(f"单独处理的层: {other_layers}")

    # 1. 首先同时处理前两个层
    if top2_layers:
        t1 = time.time()
        cloned_model = deepcopy(model_adv)
        
        # 同时启用前两个层的梯度
        for name, param in cloned_model.named_parameters():
            if name in top2_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 保存初始参数
        init_params = {}
        for layer_name in top2_layers:
            for name, param in cloned_model.named_parameters():
                if name == layer_name:
                    init_params[layer_name] = param.detach().clone()
                    break

        optimizer = torch.optim.SGD(cloned_model.parameters(), lr=args.lr_max*10, weight_decay=1e-6)

        max_target_hr_improvement = -1000.0
        cal_num = 0
        
        for epoch in range(1):
            cloned_model.train()
            # t2 = time.time()

            for ui, ii, lbl in trainloader:
                optimizer.zero_grad()
                ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                inputs_adv_u = ui
                inputs_adv_i = ii
                # inputs_adv_lab = lbl
                criterion = torch.nn.BCELoss()

                unique_users = torch.unique(inputs_adv_u).detach().clone()
                target_items_tensor = torch.tensor(target_items, device=inputs_adv_i.device)
                num_users = len(unique_users)
                num_targets = len(target_items_tensor)
                # total_pairs = num_users * num_targets

                user_seq = unique_users.repeat_interleave(num_targets)
                item_seq = target_items_tensor.repeat(num_users)
                label_seq = torch.ones_like(user_seq, dtype=torch.float)
                
                _,_,_,_,_,_,_,target_output = cloned_model(user_seq, item_seq, label_seq)
                target_yc = target_output.squeeze()
                target_losses = criterion(target_yc, label_seq)
                target_losses.backward()
                optimizer.step()
                
                if num_tcal == cal_num:
                    break
                cal_num += 1
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # 对每个层分别计算和应用扰动
            cloned_model.eval()
            sd = cloned_model.state_dict()
            
            for layer_name in top2_layers:
                diff = sd[layer_name] - init_params[layer_name]
                # times = torch.linalg.norm(diff)/torch.linalg.norm(init_params[layer_name])

                size_para = init_params[layer_name].size(0)
                normVal1 = torch.norm(diff.view(size_para, -1), 2, 1)
                normVal2 = torch.norm(init_params[layer_name].view(size_para, -1), 2, 1)
                scaling = normVal2/normVal1 * epsilon
                scaling[scaling == float('inf')] = 0
                diff = diff*scaling.view(size_para, 1)
                sd[layer_name] = deepcopy(init_params[layer_name] + diff)
            
            cloned_model.load_state_dict(sd)

            # 评估目标项目的命中率和NDCG
            t3 = time.time()
            hits10, ndcgs10, maps10, mrrs10  = evaluate_model(model_adv, testRatings, testNegatives, args.topK,topK1,topK2,topK3, num_thread=1, device=device)
            hr10, ndcg10,map10,mrr10  = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
            args.logger.info(f"Final Test: HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, mrrs10={mrr10:.8f}  [{time.time()-t1:.1f}s]")
            results = calculate_target_metrics(cloned_model, target_items, target_users,
                                        unrated_items_dict, topk_list=[10, 20, 50, 100])
            
            for metric, value in results.items():
                if metric == f"T-HR@50":
                    attacked_target_hr = value
                    args.logger.info(f"AAAAAAAAA_{metric}: {value:.8f}")
                if metric == "T-NDCG@50":
                    attacked_target_ndcg = value
                    args.logger.info(f"AAAAAAAAA_{metric}: {value:.8f}")
                args.logger.info(f"{metric}: {value:.8f}")

            # 计算命中率提升
            hr_improvement = attacked_target_hr - original_target_hr
            if hr_improvement > max_target_hr_improvement:
                max_target_hr_improvement = hr_improvement
                
        del cloned_model
        
        # 为前两个层分配相同的尖锐度值
        for layer_name in top2_layers:
            layer_sharpness_dict[layer_name[:-len(".weight")]] = max_target_hr_improvement*100
            args.logger.info("{:35}, Target HR Improvement: {:10.8f}".format(
                layer_name[:-len(".weight")], max_target_hr_improvement*100))


   # 2. 然后单独处理其他层
    for layer_name in other_layers:
        t1 = time.time()
        
        cloned_model = deepcopy(model_adv)
        for name, param in cloned_model.named_parameters():
            if name == layer_name:
                param.requires_grad = True
                init_param = param.detach().clone()
            else:
                param.requires_grad = False
        
        optimizer = torch.optim.SGD(cloned_model.parameters(), lr=args.lr_max*10, weight_decay=1e-6)

        max_target_hr_improvement = -1000.0
        cal_num = 0
        #只跑 1 个 epoch，里面用梯度上升“推高”目标物品预测分数
        for epoch in range(1):   # 这里10改3 
            # Gradient ascent   对抗攻击
            cloned_model.train()
            # t2 = time.time()

            for ui, ii, lbl in trainloader:
                optimizer.zero_grad()
                ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
                inputs_adv_u = ui
                inputs_adv_i = ii
                # inputs_adv_lab = lbl
                criterion = torch.nn.BCELoss()

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
                _,_,_,_,_,_,_,target_output = cloned_model(user_seq, item_seq, label_seq)
                target_yc = target_output.squeeze()
                # 计算目标损失：希望模型对目标项目预测为1
                target_losses = criterion(target_yc, label_seq)
                target_losses.backward()
                optimizer.step()
                if num_tcal == cal_num:
                    break
                cal_num +=1
            torch.cuda.empty_cache() if torch.cuda.is_available() else None


            cloned_model.eval()
            sd = cloned_model.state_dict()
            diff = sd[layer_name] - init_param
            # times = torch.linalg.norm(diff)/torch.linalg.norm(init_param)

            size_para = init_param.size(0)
            normVal1 = torch.norm(diff.view(size_para, -1), 2, 1)
            normVal2 = torch.norm(init_param.view(size_para, -1), 2, 1)
            scaling = normVal2/normVal1 * epsilon   # 算出一次噪声需要调整的倍数
            scaling[scaling == float('inf')] = 0
            diff = diff*scaling.view(size_para, 1)
            sd[layer_name] = deepcopy(init_param + diff)
            cloned_model.load_state_dict(sd)


                
            # 评估目标项目的命中率和NDCG
            # t3 = time.time()
            hits10, ndcgs10, maps10, mrrs10  = evaluate_model(model_adv, testRatings, testNegatives, args.topK,topK1,topK2,topK3, num_thread=1, device=device)
            hr10, ndcg10,map10,mrr10  = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
            args.logger.info(f"Final Test: HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, mrrs10={mrr10:.8f}  [{time.time()-t1:.1f}s]")
            results = calculate_target_metrics(cloned_model, target_items, target_users,
                                               unrated_items_dict, topk_list=[10, 20, 50, 100])
            for metric, value in results.items():
                if metric == f"T-HR@50":
                        attacked_target_hr = value
                        args.logger.info(f"AAAAAAAAA_{metric}: {value:.8f}")
                if metric == "T-NDCG@50":
                        attacked_target_ndcg = value
                        args.logger.info(f"AAAAAAAAA_{metric}: {value:.8f}")
                args.logger.info(f"{metric}: {value:.8f}")

                
            # 计算命中率提升s
            hr_improvement = attacked_target_hr - original_target_hr
            if hr_improvement > max_target_hr_improvement:
                max_target_hr_improvement = hr_improvement
                        
        del cloned_model
        layer_sharpness_dict[layer_name[:-len(".weight")]] = max_target_hr_improvement*100
        args.logger.info("{:35}, Target HR Improvement: {:10.8f}".format(layer_name[:-len(".weight")], max_target_hr_improvement*100))

    args.logger.info(layer_sharpness_dict)
    return layer_sharpness_dict



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MLP')
    # Data/lastfm_attack/lastfm.rev_tail.json   Data/ml1m_attack/ml1m.rev_tail.json  Data/ml1m_attack/wfm_rev_2%_head.json
    parser.add_argument("--fake_users_file", nargs="?", default="Data/ml1m_attack/ml1m_2%/ml1m.wfm_rev_head_0.02.json",help="Input data path.")
    parser.add_argument("--attack_type", nargs="?", default="rev",help="attack_type.")
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # ml-1m yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--factor-alpha', default=0.005, type=float)
    parser.add_argument('--lr-max', default=0.0001, type=float)        # 0.1 这块应该是推荐模型训练学习率 RAT ml1m 0.0005
    # parser.add_argument('--lr-prox', default=0.0001, type=float)   # in_lr_no_adv
    parser.add_argument('--attack', default='fgsm', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument("--num_steps", nargs="?", default=10)  #25

    
    parser.add_argument("--decay_factor", nargs="?", default=1.0)
    parser.add_argument('--fname', default='mlp_model', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--para_target_attack', default=True, type=bool)
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument("--alpha_awp", nargs="?", default=0)  #25 控制有目标噪声大小的参数##################
    parser.add_argument('--awp-gamma', default=0.002, type=float)   # 0.001,0.005##############################要调，控制加在层上的噪声大小
    parser.add_argument('--epsilon', default=0.02, type=float)######################控制扰动大小
    parser.add_argument('--awp-warmup', default=0, type=int)############################

    parser.add_argument("--adv_type", nargs="?", default="fgsm", choices=['fgsm', 'bim', 'pgd','mim'])
    parser.add_argument("--pro_num", nargs="?", default=1, choices=[1, 25], help="1 for fgsm and 10 for bim/pgd")
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]", #  [512, 256, 128, 64, 32, 16]  [512,  128, 32]
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,help="Number of negative instances to pair with a positive instance.")
    parser.add_argument("--num_to_select", type=int, default=3,help="num_to_select")
 
    return parser.parse_args()



def main():
    All_start_time = time.time()
    args = get_args()
    args.lr_prox = args.lr_max * 10
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # 把参数转成bunch
    args_model = Bunch()


    
    if args.awp_gamma <= 0.0:
        args.awp_warmup = np.inf

    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    suffix = 'alpha_awp{}_lr={}_epochs={}_adv_in_autogamma{}_awp-warmup={}'.format(args.alpha_awp, args.lr_max ,args.epochs,args.seed,args.awp_warmup)
    model_save_dir = './results/MLP/{}/{}/epsilon{}/{}_{}_{}_gamma{}/checkpoint/'.format(args.dataset, args.attack_type, args.epsilon, args.model, args.dataset, args.num_to_select,args.awp_gamma) + suffix + "/"

    for path in [model_save_dir]:
        if not os.path.isdir(path):
            os.makedirs(path)

    logger = create_logger(model_save_dir+'output.log')
    logger.info(args)

    for key, value in args.__dict__.items():
        setattr(args_model, key, value)
    for key, value in args_model.items():
        logger.info(f"{key}: {value}")
    logger.info('--'*50)

    args.topK = topK
    args.topK1 = topK1
    args.topK2 = topK2
    args.topK3 = topK3 
    args.device = device
    args.logger = logger

    
    # Loading data
    t1 = time.time()
    dataset = Dataset(args.data_path + args.dataset, fake_users_file=args.fake_users_file)   #######  
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    nUsers, nItems = train.shape
    args.testRatings = testRatings
    args.testNegatives = testNegatives
    args.train = train
    args.Dataset = dataset
    target_items = dataset.target_items 
    # target_items =  {1728, 2319, 1908, 1881, 2526}
    logger.info("target_items:{}".format( target_items))
    
    
    userMatrix = torch.Tensor(get_train_matrix(train))
    
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)
    logger.info(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    logger.info("MLP_adv:*************************************")
    if args.model == 'MLP':
        model_adv = MLP(fcLayers, userMatrix, itemMatrix,device)
        proxy_adv = MLP(fcLayers, userMatrix, itemMatrix,device)
    else:
        raise ValueError("Unknown model")

    model_adv = nn.DataParallel(model_adv).to(device)
    proxy_adv = nn.DataParallel(proxy_adv).to(device)

    if args.l2:
        decay, no_decay = [], []
        for name,param in model_adv.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model_adv.parameters()

    opt = torch.optim.Adam(params, lr=args.lr_max)
    proxy_opt = torch.optim.Adam(proxy_adv.parameters(), lr=args.lr_prox)   # 0.01

    args.optimizer = opt

    # temp_awp_gamma = [args.awp_gamma,args.awp_gamma,args.awp_gamma,args.awp_gamma,args.awp_gamma,args.awp_gamma,args.awp_gamma,args.awp_gamma,args.awp_gamma]
    #存储当前每层噪声大小
    temp_awp_gamma = [args.awp_gamma,args.awp_gamma,0,0,0,0,0,0,0]

    conbin_hr_best = 0.0
    modelPath = ""

    for epoch in range(args.epochs):
        start_time = time.time()
        # Generate training instances
        userInput, itemInput, labels = get_train_instances(train, args.nNeg)
        dst = BatchDataset(userInput, itemInput, labels)
        ldr = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)
        losses = AverageMeter("Loss")
        # params = model_adv.parameters()
        # opt = torch.optim.Adam(params, lr=args.lr_max)

        if epoch >= args.awp_warmup:
            temp_awp_gamma = find_gamma(args,logger, model_adv,ldr,temp_awp_gamma)
            # temp_awp_gamma = temp_awp_gamma
            logger.info("args.awp_gamma_temp{},args.pro_num: {}".format( temp_awp_gamma,args.pro_num))
            awp_adversary = AdvWeightPerturb_con(model=model_adv, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=temp_awp_gamma)
            
        num_batch = 0
        for ui, ii, lbl in ldr:
            if args.eval:
                break
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            model_adv.train()
            # calculate adversarial weight perturbation and perturb it
            if epoch >= args.awp_warmup:
                if args.para_target_attack and num_batch %20 == 0 :
                    # 随机选取5个目标项目    这里随机选取的目的是什么xq1112?  为了制作有目标噪声xq1113
                    # 使用所有项目范围（0到nItems-1）进行随机选择
                    ran_target_items = random.sample(range(nItems), min(5, nItems))
                    # 确保target_items是列表格式
                    ran_target_items = list(ran_target_items)
                    # logger.info(f"Randomly selected target items for attack: {ran_target_items}")
                awp = awp_adversary.calc_awp(inputs_adv_u=ui, inputs_adv_i=ii,inputs_adv_lab=lbl,
                                                target_items=ran_target_items,alpha_awp=args.alpha_awp)     
                awp_adversary.perturb(awp)
            robust_loss,_,_,_,_,_,_,_= model_adv(ui, ii,lbl)
            num_batch += 1
            if args.l1:
                for name,param in model_adv.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            losses.update(robust_loss.item(), lbl.size(0))
            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp)
                # 及时释放不再使用的张量
            del ui, ii, lbl, robust_loss
            torch.cuda.empty_cache()


            # Check  performance
        t1 = time.time()
        hits10, ndcgs10, maps10, mrrs10  = evaluate_model(model_adv, testRatings, testNegatives, args.topK,topK1,topK2,topK3, num_thread=1, device=device)
        hr10, ndcg10,map10,mrr10  = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        logger.info(f"Epoch {epoch}:Loss={losses.avg:.8f} [{t1-start_time:.1f}s] HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, mrrs10={mrr10:.8f}  [{time.time()-t1:.1f}s]")

        t1 = time.time()
        # 评估目标项目的命中率和NDCG
        results = calculate_target_metrics(model_adv, dataset.target_items, dataset.target_users,
                                            dataset.unrated_items_dict, topk_list=[10, 20, 50, 100])
        for metric, value in results.items():
            if metric == f"T-HR@50":
                target_hr = value   
            if metric == "T-NDCG@50":
                target_ndcg = value
            logger.info(f"{metric}: {value:.8f}")
        logger.info(f"Training {epoch}_ Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")
        logger.info("------------------------------------------------------------------------------")

            # temp_con_hr = hr10_R + hr10
        if hr10 >=  conbin_hr_best:
            conbin_hr_best = hr10  
            modelPath = f"pretrained/{args.dataset}_{args.attack_type}_MLP_TWP_{args.seed}_select{args.num_to_select}_gamma{args.awp_gamma}_awp-warmup{args.awp_warmup}_epoch{epoch}_{time.time()}.pth"
            best_model_path = modelPath  # 保存最佳模型路径

                    # 检查文件是否存在，如果存在则删除
            if os.path.exists(modelPath):
                try:
                    os.remove(modelPath)
                    logger.info(f"Deleted existing model file: {modelPath}")
                except Exception as e:
                    logger.error(f"Error deleting existing model file {modelPath}: {e}")

            os.makedirs("pretrained", exist_ok=True)
            torch.save(model_adv.state_dict(), modelPath)
            logger.info(modelPath)
            logger.info("----------")


    model_adv.eval()
            
    # 训练结束后，加载最佳模型并进行最终测试
    logger.info("="*50)
    logger.info("Training completed. Loading best model for final evaluation...")
    logger.info("="*50)
    
    if best_model_path is not None and os.path.exists(best_model_path):
        # 加载最佳模型
        model_adv.load_state_dict(torch.load(best_model_path))
        model_adv.eval()
        
        logger.info(f"Loaded best model from: {best_model_path}")
        logger.info(f"Best model HR@10: {conbin_hr_best:.8f}")
        
        # 进行最终测试
        t1 = time.time()
        hits10, ndcgs10, maps10, mrrs10  = evaluate_model(model_adv, testRatings, testNegatives, args.topK,topK1,topK2,topK3, num_thread=1, device=device)
        hr10, ndcg10,map10,mrr10  = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        logger.info(f"Final Test: HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, mrrs10={mrr10:.8f}  [{time.time()-t1:.1f}s]")

        t1 = time.time()
        # 评估目标项目的命中率和NDCG
        results = calculate_target_metrics(model_adv, dataset.target_items, dataset.target_users,
                                        dataset.unrated_items_dict, topk_list=[10, 20, 50, 100])
        for metric, value in results.items():
            if metric == f"T-HR@50":
                    target_hr = value   
            if metric == "T-NDCG@50":
                    target_ndcg = value
            logger.info(f"Final Test {metric}: {value:.8f}")
        logger.info(f"Final Test: Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")
    else:
        logger.warning("No best model found or saved. Skipping final evaluation.")
        
    # 打印最大显存使用情况
    logger.info(f"Total training time: {time.time()-All_start_time:.1f}s")




##确定每层的噪声幅度
def find_gamma(args, logger,model_adv,ldr,temp_awp_gamma):

    layer_sharpness_dict = layer_sharpness(args, model_adv, args.train, args.testRatings, args.testNegatives , args.topK, args.topK1, args.topK2, args.topK3,ldr)
    # layer_sharpness_dict = layer_sharpness(args, model_adv, args.train,ldr)
    layer_sharpness_dict111 = list(layer_sharpness_dict.items())
    tensors = [torch.tensor(item[1], device=device) for item in layer_sharpness_dict111]
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
    selected_indices = torch.multinomial(probabilities, num_to_select, replacement=False).tolist()
    
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
    logger.info(f"Randomly selected {num_to_select} layers: {selected_indices}")
    logger.info(f"Original gamma values: {temp_awp_gamma}")
    logger.info(f"Selected gamma values: {new_temp_awp_gamma}")
    logger.info(f"Selection probabilities: {probabilities.tolist()}")


    return new_temp_awp_gamma

if __name__ == "__main__":
    main()
