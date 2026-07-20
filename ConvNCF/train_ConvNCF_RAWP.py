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
import utils_convNCF

import load_dataset
import time
import random
device = torch.device("cuda:2")
torch.cuda.set_device(2)

def parse_args():
    parser = argparse.ArgumentParser(description="Run ConvNCF_RAWP.")
    # parser.add_argument("--epsilon", nargs="?", default=0.5)

    parser.add_argument("--data_path", nargs="?", default="Data/",
                        help="Input data path.")
    parser.add_argument("--dataset", nargs="?", default="ml-1m",   # lastfm yelp  AToy lastfm
                        help="Choose a dataset.")
    parser.add_argument("--fake_users_file", nargs="?", default="Data/ml1m_attack/ml1m_2%/ml1m.random_head_0.02.json",
                        help="Input data path.")
    parser.add_argument('--attack_type', default='random', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--fname', default='results/ConvNCF/RAWP', type=str)
    parser.add_argument('--awp-gamma', default=0.002,type=float)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--lr', default=0.5, type=float)

    return parser.parse_args()


class ConvNCF(nn.Module):

    def __init__(self, user_count, item_count):
        super(ConvNCF, self).__init__()

        # some variables
        self.user_count = user_count
        self.item_count = item_count

        # embedding setting
        self.embedding_size = 64

        # init target matrix of matrix factorization
        self.P = nn.Embedding(self.user_count, self.embedding_size).to(device)
        self.Q = nn.Embedding(self.item_count, self.embedding_size).to(device)

        # cnn setting
        self.channel_size = 32
        self.kernel_size = 2
        self.strides = 2
        self.cnn = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(1, self.channel_size,
                      self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 32 * 32
            nn.Conv2d(self.channel_size, self.channel_size,
                      self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 16 * 16
            nn.Conv2d(self.channel_size, self.channel_size,
                      self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size,
                      self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 4 * 4
            nn.Conv2d(self.channel_size, self.channel_size,
                      self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 2 * 2
            nn.Conv2d(self.channel_size, self.channel_size,
                      self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        )

        # fully-connected layer, used to predict
        self.fc = nn.Linear(32, 1)

    def forward(self, user_ids, item_ids, is_pretrain):

        # convert float to int
        user_ids = list(map(int, user_ids))
        item_ids = list(map(int, item_ids))

        # get embeddings, simplify one-hot to index directly
        user_embeddings = self.P(torch.tensor(user_ids).to(device))
        item_embeddings = self.Q(torch.tensor(item_ids).to(device))

#         # inner product

        if is_pretrain:
            # inner product
            prediction = torch.sum(
                torch.mul(user_embeddings, item_embeddings), dim=1)
        else:
            # outer product
            interaction_map = torch.bmm(
                user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
            interaction_map = interaction_map.view(
                (-1, 1, self.embedding_size, self.embedding_size))

            # cnn
            feature_map = self.cnn(interaction_map)
            feature_vec = feature_map.view((-1, 32))

            # fc
            prediction = self.fc(feature_vec)
            prediction = prediction.view((-1))
        return prediction


class BPRLoss(nn.Module):

    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos_preds, neg_preds):
        distance = pos_preds - neg_preds
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))

        return loss


def update_seed():
    seed = torch.rand(9)  # 生成六个介于0到1之间的随机数
    zs = [
        float(1.0000) if x > 0.5 else float(0.0000)  # 如果x大于0.5，返回1；否则返回0
        for x in seed
    ]
    return zs


def train(model, proxy_adv, path, train_dataset, testRatings, testNegatives, target_items, target_users, unrated_items_dict):
    lr = args.lr  # 0.5
    batch_size = args.batch_size
    best_hr10, best_ndcg10, bestepoch = 0, 0, -1

    model.train()
    print("oral_lr=0.5,有学习策略,gamma0.001")
    logger.info('device: {}'.format(device))
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=1e-2)
    proxy_opt = optim.Adagrad(proxy_adv.parameters(),lr=0.01, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

    train_loader = Data.DataLoader(train_dataset.train_group, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"Current GPU: {torch.cuda.current_device()}")
    bpr_loss = BPRLoss().to(device)

    for epoch in range(args.epochs):

        if epoch >= 100:
            seeds = update_seed()
            T = random.uniform(0.8, 1.2)
            awp_gamma_temp = args.awp_gamma * T
            awp_adversary = utils_convNCF.AdvWeightPerturb_convNCF_RAWP(model=model, proxy=proxy_adv, proxy_optim=proxy_opt,
                                                                        gamma=awp_gamma_temp, device=device)

            logger.info(f"Seeds: {seeds}, AWP gamma temp: {awp_gamma_temp}")

        for train_data in train_loader:
            train_data = train_data.to(device)
            user_ids = train_data[:, 0].to(device)
            pos_item_ids = train_data[:, 1].to(device)
            neg_item_ids = train_data[:, 2].to(device)

            optimizer.zero_grad()

            if epoch < 100:    # 原来100
                # pretrain bpr
                pos_preds = model(user_ids, pos_item_ids, True)
                neg_preds = model(user_ids, neg_item_ids, True)
            else:
                awp = awp_adversary.calc_awp(user_ids=user_ids, pos_item_ids=pos_item_ids, neg_item_ids=neg_item_ids)
                awp_adversary.perturb(awp, seeds)
                # # train convncf
                pos_preds = model(user_ids, pos_item_ids, False)
                neg_preds = model(user_ids, neg_item_ids, False)

            loss = bpr_loss(pos_preds, neg_preds)

            loss.backward()
            optimizer.step()

            if epoch >= 100:
                awp_adversary.restore(awp, seeds)

        scheduler.step()

        if epoch == 5 or epoch >= 99:
            t1 = time.time()

            hits10, ndcgs10, maps10, mrrs10 = utils_convNCF.evaluate_model(
                model, testRatings, testNegatives, 10, 20, 50, 100, 1, device)
            hr10, ndcg10, map10, mrr10 = np.array(hits10).mean(), np.array(
                ndcgs10).mean(), np.array(maps10).mean(), np.array(mrrs10).mean()

            logger.info(f"epoch {epoch}: HR10={hr10:.4f}, NDCG10={ndcg10:.4f}, mrrs10={mrr10:.4f} [{time.time()-t1:.1f}s]")
            if hr10 > best_hr10:
                best_hr10 = hr10
                bestepoch = epoch
                best_ndcg10 = ndcg10

                os.makedirs(os.path.join(path, 'pretrained'), exist_ok=True)
                best_model_path = os.path.join(path, 'pretrained', f"{args.dataset}_ConvNCF_{args.attack_type}_{time.time()}.pth")
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

        results = utils_convNCF.calculate_target_metrics(
            model,
            target_items,
            target_users,
            unrated_items_dict,
            topk_list=[10, 20, 50, 100],
            device=device,
        )

        hits10, ndcgs10, maps10, mrrs10 = utils_convNCF.evaluate_model(
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
    logger.info('Data Train_loading...')
    train_dataset = load_dataset.Load(args.data_path + args.dataset, args.fake_users_file, args.attack_type)
    logger.info('Data loaded')
    dataset = Dataset(args.data_path + args.dataset,
                      fake_users_file=args.fake_users_file)
    _, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    logger.info('=' * 50)

    logger.info('Model initializing...')
    model = ConvNCF(int(max(train_dataset.train_group[:, 0])) + 1, int(
        max(train_dataset.train_group[:, 1])) + 1).to(device)
    proxy_adv = ConvNCF(int(max(train_dataset.train_group[:, 0])) + 1, int(
        max(train_dataset.train_group[:, 1])) + 1).to(device)

    # awp_adversary = AdvWeightPerturb1(model=model, proxy=proxy_adv, proxy_optim=proxy_opt, gamma=awp_gamma)
    logger.info('Model initialized')

    logger.info('=' * 50)

    logger.info('Model training...')
    # train(awp_gamma)
    train(model, proxy_adv, path, train_dataset, testRatings, testNegatives, target_items=dataset.target_items,
          target_users=dataset.target_users, unrated_items_dict=dataset.unrated_items_dict)
    logger.info('Model trained')
