import sys
sys.path.append('./AT_AWP_CWB0826')
from Dataset import Dataset
import utils
import argparse

# import utils_convNCF
import numpy as np

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as Data

# 在导入utils_convNCF之前添加验证
import utils_LightGCN
from LightGCN import LightGCN

import load_dataset
import time
import random
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run LightGCN_RAWP.")
    # parser.add_argument("--epsilon", nargs="?", default=0.5)

    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # lastfm yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument("--fake_users_file", nargs="?", default="Data/ml1m_attack/ml1m_2%/ml1m.random_head_0.02.json",
                        help="Input data path.")
    parser.add_argument('--attack_type', default='random', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--fname', default='results/LightGCN/RAWP', type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--decay', default=1e-4, type=float)

    # Model Params
    parser.add_argument('--latent_dim', default=64, type=int)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--keep_prob', default=1.0, type=float) # RAWP时建议keep_prob设为1.0，避免dropout干扰扰动计算

    # AWP Params
    parser.add_argument('--awp_gamma', default=0.001, type=float)
    parser.add_argument('--awp_start_epoch', default=20, type=int, help="从第几轮开始加AWP，LightGCN前期不稳定，建议晚点加")

    return parser.parse_args()


def update_seed():
    seed = torch.rand(9, device=device)  # 生成六个介于0到1之间的随机数
    zs = [
        float(1.0000) if x > 0.5 else float(0.0000)  # 如果x大于0.5，返回1；否则返回0
        for x in seed
    ]
    return zs


def train(args, model, proxy_adv, path, train_dataset, testRatings, testNegatives, target_items, target_users, unrated_items_dict):
    lr = args.lr  # 0.5
    batch_size = args.batch_size
    best_hr10, best_ndcg10, bestepoch = 0, 0, -1
    best_model_path = ""

    # 主模型使用 Adam (LightGCN 标准)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Proxy 模型使用 Adagrad (用于计算扰动步长，通常 lr 大一点)
    proxy_opt = optim.Adagrad(proxy_adv.parameters(), lr=0.01, weight_decay=args.decay)

    train_loader = Data.DataLoader(train_dataset.train_group, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    args.logger.info(f"Start Training LightGCN-RAWP: lr={lr}, decay={args.decay}, awp_gamma={args.awp_gamma}")

    awp_adversary = None

    for epoch in range(args.epochs):

        if epoch >= args.awp_start_epoch:
            seeds = update_seed()
            T = random.uniform(0.8, 1.2)
            awp_gamma_temp = args.awp_gamma * T

            # 初始化 AWP 对象
            awp_adversary = utils_LightGCN.AdvWeightPerturb_LightGCN_RAWP(
                model=model, 
                proxy=proxy_adv, 
                proxy_optim=proxy_opt,
                gamma=awp_gamma_temp, 
                device=device
            )
            # 仅在每个epoch开始打印一次
            if epoch == args.awp_start_epoch:
                args.logger.info("RAWP Attack Started!")

        for train_data in train_loader:
            train_data = train_data.to(device)
            user_ids = train_data[:, 0].to(device)
            pos_item_ids = train_data[:, 1].to(device)
            neg_item_ids = train_data[:, 2].to(device)

            optimizer.zero_grad()

            # --- AWP 扰动过程 ---
            awp = None
            if epoch >= args.awp_start_epoch and awp_adversary is not None:
                # 1. 计算扰动方向 (Maximizing BPR Loss)
                awp = awp_adversary.calc_awp(user_ids, pos_item_ids, neg_item_ids)
                # 2. 将扰动加到主模型上
                awp_adversary.perturb(awp, seeds)

            loss = model(user_ids, pos_item_ids, neg_item_ids, decay=args.decay)
            loss.backward()
            optimizer.step()

            if epoch >= args.awp_start_epoch and awp_adversary is not None and awp is not None:
                awp_adversary.restore(awp, seeds)


        t1 = time.time()

        hits10, ndcgs10, maps10, mrrs10 = utils_LightGCN.evaluate_model(model, testRatings, testNegatives, 10, 20, 50, 100, 1, device)
        hr10, ndcg10, map10, mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(maps10).mean(), np.array(mrrs10).mean()

        logger.info(f"epoch {epoch}: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f} [{time.time()-t1:.1f}s]")
        if hr10 > best_hr10:
            best_hr10 = hr10
            bestepoch = epoch
            best_ndcg10 = ndcg10

            os.makedirs(os.path.join(path, 'pretrained'), exist_ok=True)
            best_model_path = os.path.join(path, 'pretrained', f"{args.dataset}_LightGCN_RAWP_{args.attack_type}_{time.time()}.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best checkpoint to: {best_model_path} (epoch {bestepoch}: 1HR10={best_hr10:.8f})")
        torch.cuda.empty_cache()
    logger.info(f"Best epoch {bestepoch}: HR10={best_hr10:.4f}, NDCG10={best_ndcg10:.4f}")

    # Optional: final load best checkpoint and print final target HR
    if best_model_path and os.path.exists(best_model_path):
        logger.info("=" * 60)
        logger.info("Loading best checkpoint for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        results = utils_LightGCN.calculate_target_metrics(
            model,
            target_items,
            target_users,
            unrated_items_dict,
            topk_list=[10, 20, 50, 100],
            device=device,
        )

        hits10, ndcgs10, maps10, mrrs10 = utils_LightGCN.evaluate_model(
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
            logger.info(f"Final {metric}: {value:.8f}")

        logger.info(f"best_model_path: {best_model_path}")


if __name__ == '__main__':

    # torch.set_num_threads(12)
    args = parse_args()
    # Directory and logger
    path = os.path.join(args.fname, args.dataset, args.attack_type)
    os.makedirs(path, exist_ok=True)
    logger = utils.create_logger(os.path.join(path, 'output.log'))
    # torch.set_num_threads(12)
    logger.info(args)
    args.logger = logger
    logger.info('Data Train_loading...')
    train_dataset = load_dataset.Load(args.data_path + args.dataset, args.fake_users_file, args.attack_type)
    logger.info('Data loaded')
    dataset = Dataset(args.data_path + args.dataset,
                      fake_users_file=args.fake_users_file)
    _, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    user_count = int(max(train_dataset.train_group[:, 0])) + 1
    item_count = int(max(train_dataset.train_group[:, 1])) + 1
    logger.info(f"Users: {user_count}, Items: {item_count}")
    # 构建邻接矩阵
    adj_mat = utils_LightGCN.get_adj_mat(train_dataset.train_group, user_count, item_count)
    adj_mat = adj_mat.to(device)
    logger.info('=' * 50)

    logger.info('Model initializing...')
    # 1. 初始化 Main Model
    model = LightGCN(
        user_count=user_count, 
        item_count=item_count, 
        device=device, 
        adj_mat=adj_mat, 
        latent_dim=args.latent_dim, 
        n_layers=args.layers, 
        keep_prob=args.keep_prob
    ).to(device)

    proxy_adv = LightGCN(
        user_count=user_count, 
        item_count=item_count, 
        device=device, 
        adj_mat=adj_mat,  # 共享同一个图结构张量即可
        latent_dim=args.latent_dim, 
        n_layers=args.layers, 
        keep_prob=args.keep_prob
    ).to(device)


    # awp_adversary = AdvWeightPerturb1(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=awp_gamma)
    logger.info('Model initialized')

    logger.info('=' * 50)

    logger.info('Model training...')
    # train(awp_gamma)
    # 开始训练
    train(args, model, proxy_adv, path, train_dataset, testRatings, testNegatives, 
          target_items=dataset.target_items,
          target_users=dataset.target_users, 
          unrated_items_dict=dataset.unrated_items_dict)
    logger.info('Model trained')
