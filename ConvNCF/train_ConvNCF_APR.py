import sys
sys.path.append('./AT_AWP_CWB0826')  
import load_dataset
import torch
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import numpy as np
import time
from Dataset import Dataset
import utils_convNCF
import utils

# 引入 APR 模型 (假设你的模型文件名为 ConvNCF_APR.py)
from convNCF_apr import ConvNCF_APR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(description="Run ConvNCF with APR.")
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # lastfm yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument("--fake_users_file", nargs="?", default="Data/ml1m_attack/ml1m_2%/ml1m.random_head_0.02.json",
                        help="Input data path.")
    parser.add_argument('--attack_type', default='random', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--fname', default='results/ConvNCF/APR', type=str) # 修改默认保存路径区分
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', default=0.5, type=float)
    # APR 特有参数 (虽然模型类里硬编码了，但这里可以留接口后续扩展)
    parser.add_argument('--alpha', default=1.0, type=float, help="Adversarial regularization weight.")
    parser.add_argument('--epsilon', default=0.5, type=float, help="Adversarial perturbation budget.")
    parser.add_argument('--cnn_start_epoch', default=99, type=int, help="Start epoch for CNN training.")
    parser.add_argument('--adv_warmup_epochs', default=10, type=int, help="Warmup epochs for APR.")
    
    return parser.parse_args()

def train(model, path, train_dataset, testRatings, testNegatives, target_items, target_users, unrated_items_dict):
    lr = args.lr
    print(f"mine_lr={lr}, APR Training, Pretrain<99 epoch")
    epoches = args.epochs
    batch_size = args.batch_size
    best_hr10, best_ndcg10, bestepoch = 0, 0, -1
    
    # 可以在这里覆盖模型内部默认的 APR 参数
    model.alpha = args.alpha
    model.epsilon = args.epsilon
    
    model.train()

    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    train_loader = Data.DataLoader(train_dataset.train_group, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    # 注意：ConvNCF_APR 内部已经集成了 Loss 计算，不需要在此定义 BPRLoss

    # 建议设置一个 warm-up 窗口，例如 10 到 20 个 epoch
    cnn_start_epoch = args.cnn_start_epoch
    adv_warmup_epochs = args.adv_warmup_epochs  # 让 CNN 单独跑 20 个 epoch
    adv_start_epoch = cnn_start_epoch + adv_warmup_epochs # 119 epoch 开始 APR

    for epoch in range(epoches):    
        model.train() 
        
        # === 逻辑判断 ===
        if epoch < cnn_start_epoch:
            # 阶段 1: MF 预训练
            is_pretrain_stage = True
            enable_adv = False
            current_stage_name = "MF Pretrain"
        elif epoch < adv_start_epoch:
            # 阶段 2: CNN 热身 (Clean Training)
            is_pretrain_stage = False
            enable_adv = False
            current_stage_name = "CNN Warm-up"
        else:
            # 阶段 3: APR 对抗训练
            is_pretrain_stage = False
            enable_adv = True
            current_stage_name = "CNN + APR"
        
        # 可以在 log 里打印一下当前状态，确认是否生效
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Stage=[{current_stage_name}], APR={enable_adv}")

        for train_data in train_loader:  
            train_data = train_data.to(device)                  
            user_ids = train_data[:, 0].to(device)
            pos_item_ids = train_data[:, 1].to(device)
            neg_item_ids = train_data[:, 2].to(device)
                    
            optimizer.zero_grad()

            # 调用 ConvNCF_APR 的 forward
            # 它会返回 total_loss (Original + Adversarial) 以及预测值
            loss, _, _ = model(
                user_ids, 
                pos_item_ids, 
                neg_item_ids, 
                is_pretrain=is_pretrain_stage, 
                user_adv=enable_adv
            )

            loss.backward()
            
            for name, p in model.named_parameters():
                if p.grad is not None and torch.isnan(p.grad).any():
                    print("NaN grad in", name)
                    exit()
            
            optimizer.step()

        scheduler.step()
        
        # 评估部分保持不变
        if epoch == 5 or epoch >= 99:
            t1 = time.time()
            current_is_pretrain = (epoch < 99) 
            # 评估时只看 clean 性能
            hits10, ndcgs10, maps10, mrrs10 = utils_convNCF.evaluate_model_apr(model, testRatings, testNegatives, 10, 20, 50, 100, 1, device, is_pretrain_eval=current_is_pretrain)
            hr10, ndcg10, map10, mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(), np.array(maps10).mean(), np.array(mrrs10).mean()
                
            logger.info(f"epoch {epoch} [APR={enable_adv}]: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f} [{time.time()-t1:.1f}s]")

            if hr10 >= best_hr10:
                best_hr10 = hr10
                bestepoch = epoch
                best_ndcg10 = ndcg10
            
                os.makedirs(os.path.join(path, 'pretrained'), exist_ok=True)
                # 修改保存文件名以体现 APR
                best_model_path = os.path.join(path, 'pretrained', f"{args.dataset}_ConvNCF_APR_{args.attack_type}_{time.time()}.pth")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best checkpoint to: {best_model_path} (epoch {bestepoch}: HR10={best_hr10:.8f})")

        torch.cuda.empty_cache()               
    
    logger.info(f"Best epoch {bestepoch}: HR10={best_hr10:.4f}, NDCG10={best_ndcg10:.4f}")

    # Final Evaluation (保持原样)
    if best_model_path and os.path.exists(best_model_path):
        logger.info("=" * 60)
        logger.info("Loading best checkpoint for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

         # === 修改点：根据 bestepoch 动态判断使用哪种模式 ===
        final_is_pretrain = (bestepoch < 99)
        if final_is_pretrain:
            logger.warning("Best model was found during Pre-training phase (MF mode)!")
        else:
            logger.info("Best model was found during ConvNCF phase (CNN mode).")

        results = utils_convNCF.calculate_target_metrics_apr(
            model,
            target_items,
            target_users,
            unrated_items_dict,
            topk_list=[10, 20, 50, 100],
            device=device,
            is_pretrain_eval=final_is_pretrain
        )

        hits10, ndcgs10, maps10, mrrs10 = utils_convNCF.evaluate_model_apr(
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
        logger.info(f"Final Test: HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, MAP10={map10:.8f}, MRR10={mrr10:.8f}  [{time.time()-t1:.1f}s]")
        for metric, value in results.items():
            if metric == f"T-HR@50":
                    target_hr = value   
            if metric == "T-NDCG@50":
                    target_ndcg = value
            logger.info(f"Final Test {metric}: {value:.8f}")
        logger.info(f"Final Test: Target_HR50 = {target_hr:.8f}, Target_NDCG50 = {target_ndcg:.8f}, [{time.time()-t1:.1f}s]")

        logger.info(f"best_model_path: {best_model_path}")

if __name__ == '__main__':
    args = parse_args()
    # Directory and logger
    path = os.path.join(args.fname, args.dataset, args.attack_type)
    os.makedirs(path, exist_ok=True)
    logger = utils.create_logger(os.path.join(path, 'output.log'))

    logger.info("=" * 60)
    logger.info("ConvNCF with APR (Adversarial Training)")
    logger.info(args)
    logger.info(f"device: {device}")
    logger.info("=" * 60)
    
    logger.info('Data Train_loading...')
    train_dataset = load_dataset.Load(args.data_path + args.dataset, fake_file_file=args.fake_users_file, attack_type=args.attack_type)
    logger.info('Data loaded')
    dataset = Dataset(args.data_path + args.dataset, fake_users_file=args.fake_users_file)
    _, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    logger.info('=' * 50)

    logger.info('Model initializing...')
    # 实例化 ConvNCF_APR
    model = ConvNCF_APR(
        user_count=int(max(train_dataset.train_group[:, 0])) + 1, 
        item_count=int(max(train_dataset.train_group[:, 1])) + 1, 
        device=device,
        alpha=args.alpha,
        epsilon=args.epsilon
    ).to(device)
    
    logger.info('Model initialized')

    logger.info('=' * 50)

    logger.info('Model training...')
    train(model, path, train_dataset, testRatings, testNegatives, target_items=dataset.target_items,
    target_users=dataset.target_users, unrated_items_dict=dataset.unrated_items_dict)

    logger.info('Model trained')