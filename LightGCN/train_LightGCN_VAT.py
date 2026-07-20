import sys
sys.path.append('./AT_AWP_CWB0826')  
import load_dataset
import torch
import argparse
import os
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import time
import scipy.sparse as sp
from Dataset import Dataset
import utils_LightGCN

# 假设你把之前我写的 LightGCN_VAT 类放到了 LightGCN.py 文件中
# 如果类名不同或文件不同，请相应修改 import
from LightGCN_VAT import LightGCN_VAT 

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Run LightGCN with VAT.")
    parser.add_argument("--data_path", nargs="?", default="Data/", help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m", help="Choose a dataset.")
    parser.add_argument("--fake_users_file", nargs="?", default="Data/ml1m_attack/ml1m_2%/ml-1m.dp_head_0.02.json", help="Input data path.")
    parser.add_argument('--attack_type', default='dp', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--fname', default='results/LightGCN/VAT', type=str) # 修改默认保存路径
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--keep_prob', default=1.0, type=float)
    
    # --- VAT 特有参数 ---
    parser.add_argument('--alpha', default=1.0, type=float, help="对抗Loss的权重")
    parser.add_argument('--epsilon', default=0.1, type=float, help="对抗扰动的幅度")
    parser.add_argument('--user_lmb', default=2.0, type=float, help="用户自适应权重的lambda")
    parser.add_argument('--adv_epoch', default=20, type=int, help="从第几个epoch开始进行对抗训练")
    
    return parser.parse_args()

def train(args, model, path, train_dataset, testRatings, testNegatives, target_items, target_users, unrated_items_dict):
    lr = args.lr
    epoches = args.epochs
    batch_size = args.batch_size
    best_hr10, bestepoch = 0, -1
    best_model_path = ""
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = Data.DataLoader(train_dataset.train_group, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    args.logger.info(f"Start Training LightGCN_VAT: lr={lr}, decay={args.decay}, layers={args.layers}")
    args.logger.info(f"VAT Config: alpha={args.alpha}, epsilon={args.epsilon}, user_lmb={args.user_lmb}")

    for epoch in range(epoches):
        model.train()
        total_loss = 0.0
        t1 = time.time()
            # [新增] 判断当前 epoch 是否需要开启对抗训练
        # 通常建议先进行普通训练收敛一段时间，再开启 APR
        is_adv = True if epoch >= args.adv_epoch else False
        phase_name = "ADV" if is_adv else "Normal"
        for train_data in train_loader:
            user_ids = train_data[:, 0].to(device)
            pos_item_ids = train_data[:, 1].to(device)
            neg_item_ids = train_data[:, 2].to(device)
            
            optimizer.zero_grad()

            # --- 核心修改 ---
            # 调用 forward，设置 user_adv=True 开启对抗训练
            # LightGCN_VAT 内部会计算: BPR_Loss + Reg_Loss + alpha * VAT_Loss
            loss = model(user_ids, pos_item_ids, neg_item_ids, 
                         decay=args.decay, 
                         user_adv=is_adv) 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 评估 (保持原样，评估时不需要 VAT，只需用模型当前的参数进行推理)
        model.eval()
        with torch.no_grad():
            hits10, ndcgs10, maps10, mrrs10 = utils_LightGCN.evaluate_model(model, testRatings, testNegatives, 10, 20, 50, 100, 1, device)
            hr10 = np.array(hits10).mean()
            ndcg10 = np.array(ndcgs10).mean()

            results = utils_LightGCN.calculate_target_metrics(
                model, 
                target_items=target_items,
                target_users=target_users,
                unrated_items_dict=unrated_items_dict,
                topk_list=[10, 20, 50, 100],
                device=device
            )
            
        # [修改] 日志增加当前阶段显示
        args.logger.info(f"Epoch {epoch} [{phase_name}]: loss={total_loss/len(train_loader):.4f}, HR10={hr10:.4f}, NDCG10={ndcg10:.4f} [{time.time()-t1:.1f}s]")
        for metric, value in results.items():
            if metric == f"T-HR@50":
                    target_hr = value   
            if metric == "T-NDCG@50":
                    target_ndcg = value
            args.logger.info(f"Final Test {metric}: {value:.8f}")
        args.logger.info(f"Final Test: Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")
            
        if hr10 >= best_hr10:
            best_hr10 = hr10
            bestepoch = epoch
            os.makedirs(os.path.join(path, 'pretrained'), exist_ok=True)
            # 文件名加上 VAT 参数，方便区分
            model_name = f"{args.dataset}_LightGCN_VAT_{args.attack_type}_a{args.alpha}_e{args.epsilon}.pth"
            best_model_path = os.path.join(path, 'pretrained', model_name)
            torch.save(model.state_dict(), best_model_path)
            args.logger.info(f"Saved best checkpoint to: {best_model_path} (epoch {bestepoch}: HR10={best_hr10:.8f})")


    args.logger.info(f"Best epoch {bestepoch}: HR10={best_hr10:.4f}")

    # Final Evaluation
    if best_model_path and os.path.exists(best_model_path):
        args.logger.info("=" * 60)
        args.logger.info("Loading best checkpoint for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        # 目标攻击评估
        results = utils_LightGCN.calculate_target_metrics(
                model, 
                target_items=target_items,
                target_users=target_users,
                unrated_items_dict=unrated_items_dict,
                topk_list=[10, 20, 50, 100],
                device=device
            )

        # 总体推荐质量评估
        hits10, ndcgs10, maps10, mrrs10 = utils_LightGCN.evaluate_model(
                model, testRatings=testRatings, testNegatives=testNegatives, 
                        topK=10, topK1=20, topK2=50, topK3=100, device=device
                )
        hr10 = float(np.array(hits10).mean())
        ndcg10 = float(np.array(ndcgs10).mean())
        map10 = float(np.array(maps10).mean())
        mrr10 = float(np.array(mrrs10).mean())
        
        args.logger.info(f"Final Test: HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, MAP10={map10:.8f}, MRR10={mrr10:.8f}  [{time.time()-t1:.1f}s]")
        for metric, value in results.items():
            if metric == f"T-HR@50":
                    target_hr = value   
            if metric == "T-NDCG@50":
                    target_ndcg = value
            args.logger.info(f"Final Test {metric}: {value:.8f}")
        
        if 'target_hr' in locals():
            args.logger.info(f"Final Test: Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}")

        args.logger.info(f"best_model_path: {best_model_path}")

if __name__ == '__main__':
    args = parse_args()
    
    # 路径加上 parameters 标识，避免日志混淆
    sub_folder = f"{args.attack_type}_alpha{args.alpha}_eps{args.epsilon}"
    path = os.path.join(args.fname, args.dataset, sub_folder)
    
    os.makedirs(path, exist_ok=True)
    logger = utils_LightGCN.create_logger(os.path.join(path, 'output.log'))
    logger.info(args)
    args.logger = logger

    logger.info("=" * 60)
    logger.info(f"LightGCN-VAT Training Script. Dataset: {args.dataset}")
    
    logger.info('Loading Data...')
    train_dataset = load_dataset.Load(args.data_path + args.dataset, fake_file_file=args.fake_users_file, attack_type=args.attack_type)
    dataset = Dataset(args.data_path + args.dataset, fake_users_file=args.fake_users_file)
    _, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    user_count = int(max(train_dataset.train_group[:, 0])) + 1
    item_count = int(max(train_dataset.train_group[:, 1])) + 1
    logger.info(f"Users: {user_count}, Items: {item_count}")

    # 构建邻接矩阵
    adj_mat = utils_LightGCN.get_adj_mat(train_dataset.train_group, user_count, item_count)
    adj_mat = adj_mat.to(device)

    # 初始化 LightGCN_VAT 模型
    model = LightGCN_VAT(
        user_count=user_count, 
        item_count=item_count, 
        device=device, 
        adj_mat=adj_mat, 
        latent_dim=args.latent_dim, 
        n_layers=args.layers, 
        keep_prob=args.keep_prob,
        # 传入 VAT 参数
        alpha=args.alpha,
        epsilon=args.epsilon,
        user_lmb=args.user_lmb
    ).to(device)

    train(args, model, path, train_dataset, testRatings, testNegatives, 
          target_items=dataset.target_items,
          target_users=dataset.target_users, 
          unrated_items_dict=dataset.unrated_items_dict)