
import argparse
import time
from bunch import Bunch

import numpy as np
import torch
import torch.nn as nn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from mlp import MLP  
from Dataset import Dataset


from evaluate import *
from utils import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='MLP_clean')
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # ml-1m yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument("--fake_users_file", nargs="?", default="Data/ml1m_attack/ml1m_3%/ml-1m.dp_head_0.03.json",help="Input fake users file.")
    parser.add_argument("--attack_type", nargs="?", default="dp_003",help="attack_type.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr-max', default=0.0001, type=float)        # 0.1 这块应该是推荐模型训练学习率 RAT ml1m 0.0005
    parser.add_argument('--fname', default='results/MLP/clean', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument("--fcLayers", nargs="?", default="[1024, 512, 256, 128, 64, 32, 16]", #  [512, 256, 128, 64, 32, 16]  [512,  128, 32]
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So fcLayers[0]/2 is the embedding size.")
    parser.add_argument("--nNeg", type=int, default=4,help="Number of negative instances to pair with a positive instance.")
 
    return parser.parse_args()


def main():
    All_start_time = time.time()
    args = get_args()
    path = os.path.join(args.fname, args.dataset, args.attack_type)
    os.makedirs(path, exist_ok=True)
    logger = utils.create_logger(os.path.join(path, 'output.log'))
    
    fcLayers = eval(args.fcLayers)
    topK = 10
    topK1 = 20
    topK2 = 50
    topK3 = 100
    args.topK = topK
    args.topK1 = topK1
    args.topK2 = topK2
    args.topK3 = topK3 
    args.device = device
    args.logger = logger
    
    # 把参数转成bunch
    args_model = Bunch()
    for key, value in args.__dict__.items():
        setattr(args_model, key, value)
    for key, value in args_model.items():
        print(f"{key}: {value}")
    print('--'*50)

    logger.info(args)
    
    # Loading data
    t1 = time.time()
    logger.info('Data Train_loading...')
    dataset = Dataset(args.data_path + args.dataset, args.fake_users_file)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    nUsers, nItems = train.shape
    args.testRatings = testRatings
    args.testNegatives = testNegatives
    args.train = train
    args.Dataset = dataset
    target_items = dataset.target_items 
    logger.info("target_items:{}".format(target_items))
    
    userMatrix = torch.Tensor(get_train_matrix(train))
    
    itemMatrix = torch.transpose(torch.Tensor(get_train_matrix(train)), 0, 1)
    userMatrix, itemMatrix = userMatrix.to(device), itemMatrix.to(device)
    logger.info(f"Load data: #user={nUsers}, #item={nItems}, #train={train.nnz}, #test={len(testRatings)} [{time.time()-t1:.1f}s]")
    logger.info("MLP_clean:*************************************")
    if args.model == 'MLP_clean':
        model = MLP(fcLayers, userMatrix, itemMatrix, device)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    best_hr10 = -1
    bestepoch = -1
    best_ndcg10 = -1
        
    for epoch in range(args.epochs):
        start_time = time.time()
        # Generate training instances
        userInput, itemInput, labels = get_train_instances(train, args.nNeg)

        dst = BatchDataset(userInput, itemInput, labels)
        ldr = torch.utils.data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, drop_last=True)
        losses = AverageMeter("Loss")
        model.train()
        for ui, ii, lbl in ldr:
            if args.eval:
                break
            ui, ii, lbl = ui.to(device), ii.to(device), lbl.to(device)  
            robust_loss,_,_,_,_,_,_,_= model(ui, ii,lbl)
            opt.zero_grad()
            robust_loss.backward()
            opt.step()
            losses.update(robust_loss.item(), lbl.size(0))

        
        # Check  performance
        t1 = time.time()
        hits10, ndcgs10, maps10, mrrs10= evaluate_model(model, testRatings, testNegatives, topK,topK1,topK2,topK3, num_thread=1, device=device)
        hr10, ndcg10,map10,mrr10 = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        logger.info(f"Epoch {epoch}:Loss={losses.avg:.4f} [{t1-start_time:.1f}s] HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f} [{time.time()-t1:.1f}s]")
        
        results = calculate_target_metrics(model, dataset.target_items, dataset.target_users,
                                        dataset.unrated_items_dict, topk_list=[10, 20, 50, 100])
        for metric, value in results.items():
            if metric == f"T-HR@50":
                    target_hr = value   
            if metric == "T-NDCG@50":
                    target_ndcg = value
            logger.info(f"Test_THR {metric}: {value:.8f}")

       
        if hr10 > best_hr10:
            best_hr10 = hr10
            bestepoch = epoch
            best_ndcg10 = ndcg10

            os.makedirs(os.path.join(path, 'pretrained'), exist_ok=True)
            best_model_path = os.path.join(path, 'pretrained', f"{args.dataset}_MLP_{args.attack_type}_{time.time()}.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best checkpoint to: {best_model_path} (epoch {bestepoch}: 1HR10={best_hr10:.8f})")
        torch.cuda.empty_cache()
    
    logger.info(f"Best HR10={best_hr10:.4f}, NDCG10={best_ndcg10:.4f}")
    
    logger.info(f"Total time: {time.time()-All_start_time:.1f}s")
     # 训练结束后，加载最佳模型并进行最终测试
    logger.info("="*50)
    logger.info("Training completed. Loading best model for final evaluation...")
    logger.info("="*50)
    if best_model_path is not None and os.path.exists(best_model_path):
        # 加载最佳模型
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        logger.info(f"Loaded best model from: {best_model_path}")
        logger.info(f"Best model HR@10: {best_hr10:.8f}")
        
        # 进行最终测试
        t1 = time.time()
        hits10, ndcgs10, maps10, mrrs10  = evaluate_model(model, testRatings, testNegatives, args.topK,topK1,topK2,topK3, num_thread=1, device=device)
        hr10, ndcg10,map10,mrr10  = np.array(hits10).mean(), np.array(ndcgs10).mean(),np.array(maps10).mean(), np.array(mrrs10).mean()
        logger.info(f"Final Test: HR10={hr10:.8f}, NDCG10={ndcg10:.8f}, mrrs10={mrr10:.8f}  [{time.time()-t1:.1f}s]")

        t1 = time.time()
        # 评估目标项目的命中率和NDCG
        results = calculate_target_metrics(model, dataset.target_items, dataset.target_users,
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


if __name__ == "__main__":
    main()
