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

from LightGCN import LightGCN  # 导入 LightGCN 类


# 导入你上面给出的 LightGCN 类
# 如果是在同一个文件里直接写类名，如果在不同文件请 from your_file import LightGCN
# from model_file import LightGCN 

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Run LightGCN.")
    parser.add_argument("--data_path", nargs="?", default="Data/", help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m", help="Choose a dataset.")
    parser.add_argument("--fake_users_file", nargs="?", default="Data/ml1m_attack/ml1m_2%/ml1m.random_head_0.02.json", help="Input data path.")
    parser.add_argument('--attack_type', default='random', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--fname', default='results/LightGCN/clean', type=str)
    parser.add_argument('--batch_size', default=1024, type=int) # GCN通常batch可以大一点
    parser.add_argument('--lr', default=0.001, type=float) # LightGCN标准学习率
    parser.add_argument('--decay', default=1e-4, type=float) # 正则化系数
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--keep_prob', default=1.0, type=float)
    
    return parser.parse_args()

def train(args, model, path, train_dataset, testRatings, testNegatives, target_items, target_users, unrated_items_dict):
    lr = args.lr
    epoches = args.epochs
    batch_size = args.batch_size
    best_hr10, bestepoch = 0, -1
    best_model_path = ""
    
    # LightGCN 官方通常推荐 Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_loader = Data.DataLoader(train_dataset.train_group, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    args.logger.info(f"Start Training LightGCN: lr={lr}, decay={args.decay}, layers={args.layers}")

    for epoch in range(epoches):
        model.train()
        total_loss = 0.0
        t1 = time.time()
        
        for train_data in train_loader:
            user_ids = train_data[:, 0].to(device)
            pos_item_ids = train_data[:, 1].to(device)
            neg_item_ids = train_data[:, 2].to(device)
            
            optimizer.zero_grad()

            # 调用你提供的 LightGCN forward 接口
            # 该接口内部已经处理了 BPR Loss + decay * Reg Loss
            loss = model(user_ids, pos_item_ids, neg_item_ids, decay=args.decay)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 评估
        
        model.eval()
        with torch.no_grad():
            # 注意：utils_convNCF.evaluate_model 内部需要支持调用 model.predict(u, i)
            hits10, ndcgs10, maps10, mrrs10 = utils_LightGCN.evaluate_model(model, testRatings, testNegatives, 10, 20, 50, 100, 1, device)
            hr10 = np.array(hits10).mean()
            ndcg10 = np.array(ndcgs10).mean()
        args.logger.info(f"epoch {epoch}: loss={total_loss/len(train_loader):.4f}, HR10={hr10:.4f}, NDCG10={ndcg10:.4f} [{time.time()-t1:.1f}s]")
            
        if hr10 >= best_hr10:
            best_hr10 = hr10
            bestepoch = epoch
            os.makedirs(os.path.join(path, 'pretrained'), exist_ok=True)
            best_model_path = os.path.join(path, 'pretrained', f"{args.dataset}_LightGCN_{args.attack_type}.pth")
            torch.save(model.state_dict(), best_model_path)
            args.logger.info(f"Saved best checkpoint to: {best_model_path} (epoch {bestepoch}: HR10={best_hr10:.8f})")


    args.logger.info(f"Best epoch {bestepoch}: HR10={best_hr10:.4f}")

    # Final Evaluation (保持原样)
    if best_model_path and os.path.exists(best_model_path):
        args.logger.info("=" * 60)
        args.logger.info("Loading best checkpoint for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        results = utils_LightGCN.calculate_target_metrics(
                model, 
                target_items=target_items,
                target_users=target_users,
                unrated_items_dict=unrated_items_dict,
                topk_list=[10, 20, 50, 100],
                device=device
            )

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
        args.logger.info(f"Final Test: Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")

        args.logger.info(f"best_model_path: {best_model_path}")

if __name__ == '__main__':
    args = parse_args()
    path = os.path.join(args.fname, args.dataset, args.attack_type)
    os.makedirs(path, exist_ok=True)
    logger = utils_LightGCN.create_logger(os.path.join(path, 'output.log'))
    args.logger = logger

    logger.info("=" * 60)
    logger.info(f"LightGCN Training Script. Dataset: {args.dataset}")
    
    # 加载数据 (沿用你的 load_dataset)
    logger.info('Loading Data...')
    train_dataset = load_dataset.Load(args.data_path + args.dataset, fake_file_file=args.fake_users_file, attack_type=args.attack_type)
    dataset = Dataset(args.data_path + args.dataset, fake_users_file=args.fake_users_file)
    _, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives

    user_count = int(max(train_dataset.train_group[:, 0])) + 1
    item_count = int(max(train_dataset.train_group[:, 1])) + 1
    logger.info(f"Users: {user_count}, Items: {item_count}")

    # 构建邻接矩阵 (LightGCN 核心步骤)
    adj_mat = utils_LightGCN.get_adj_mat(train_dataset.train_group, user_count, item_count)
    adj_mat = adj_mat.to(device)

    # 初始化模型
    model = LightGCN(
        user_count=user_count, 
        item_count=item_count, 
        device=device, 
        adj_mat=adj_mat, 
        latent_dim=args.latent_dim, 
        n_layers=args.layers, 
        keep_prob=args.keep_prob
    ).to(device)

    # 开始训练
    train(args, model, path, train_dataset, testRatings, testNegatives, 
          target_items=dataset.target_items,
          target_users=dataset.target_users, 
          unrated_items_dict=dataset.unrated_items_dict)