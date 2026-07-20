import numpy as np

import torch
import torch.nn as nn

import time



class MLP(nn.Module):
    def __init__(self, fcLayers, userMatrix, itemMatrix, device):
        super(MLP, self).__init__()
        self.device = device
        self.register_buffer("userMatrix", userMatrix)
        self.register_buffer("itemMatrix", itemMatrix)
        nUsers = self.userMatrix.size(0)
        nItems = self.itemMatrix.size(0)
        self.criterion = torch.nn.BCELoss()

        # In the official implementation,
        # the first dense layer has no activation
        self.userFC = nn.Linear(nItems, fcLayers[0] // 2)
        self.itemFC = nn.Linear(nUsers, fcLayers[0] // 2)
        layers = []
        for l1, l2 in zip(fcLayers[:-1], fcLayers[1:]):
            layers.append(nn.Linear(l1, l2))
            layers.append(nn.ReLU(inplace=False))
        self.fcs = nn.Sequential(*layers)

        # In the official implementation,
        # the final module is initialized using Lecun normal method.
        # Here, the Kaiming normal initialization is adopted.
        self.final = nn.Sequential(
            nn.Linear(fcLayers[-1], 1),
            nn.Sigmoid(),
        )

    def forward(self, user, item, label):
        userInput = self.userMatrix[user, :].to(self.device)  # (B, 3706)
        itemInput = self.itemMatrix[item, :].to(self.device)  # (B, 6040)
        userVector = self.userFC(userInput).to(self.device)  # (B, fcLayers[0]//2)
        itemVector = self.itemFC(itemInput).to(self.device)  # (B, fcLayers[0]//2)
        self.y0 = torch.cat((userVector, itemVector), -1).to(self.device)
        self.y1 = self.fcs[0](self.y0).to(self.device)
        self.y1.requires_grad_(True)  # 241029 跑干净MLP的时候，这里总报错，所以暂时注释掉
        self.y1.retain_grad()
        self.y2 = self.fcs[1](self.y1)

        # # 获取 self.fcs[1] 中每个子层的参数
        # for idx, layer in enumerate(self.fcs.children()):
        #     print(f"Layer {idx}:")
        #     for name, param in layer.named_parameters():
        #         print(f"Parameter name: {name}, Value: {param}")

        self.y3 = self.fcs[2](self.y2).to(self.device)
        self.y3.requires_grad_(True)
        self.y3.retain_grad()
        self.y4 = self.fcs[3](self.y3)

        self.y5 = self.fcs[4](self.y4).to(self.device)
        self.y5.requires_grad_(True)
        self.y5.retain_grad()
        self.y6 = self.fcs[5](self.y5)

        self.y7 = self.fcs[6](self.y6).to(self.device)
        self.y7.requires_grad_(True)
        self.y7.retain_grad()
        self.y8 = self.fcs[7](self.y7)

        self.y9 = self.fcs[8](self.y8).to(self.device)
        self.y9.requires_grad_(True)
        self.y9.retain_grad()
        self.y10 = self.fcs[9](self.y9)

        self.y11 = self.fcs[10](self.y10).to(self.device)
        self.y11.requires_grad_(True)
        self.y11.retain_grad()
        self.y12 = self.fcs[11](self.y11)
        # (B, fcLayers[-1])
        y = self.final(self.y12).to(self.device)
        # (B, fcLayers[-1])                          # (B, 1)
        yc = y.squeeze()
        loss = self.criterion(yc, label)
        return loss, self.y1, self.y3, self.y5, self.y7, self.y9, self.y11, y

    def choose_layer(self):
        # if self.enable_lat == False:
        #     return
        if self.layerlist == 'all':
            self.enable_list1 = list(range(0, self.seed1 + 1))
            self.enable_list2 = list(range(0, self.seed2 + 1))
            self.enable_list3 = list(range(0, self.seed3 + 1))
            self.enable_list4 = list(range(0, self.seed4 + 1))
            self.enable_list5 = list(range(0, self.seed5 + 1))
            self.enable_list6 = list(range(0, self.seed6 + 1))  # all True
        else:
            for i in self.layerlist_digit:
                self.enable_list[i] = 1

    def update_seed(self):
        self.seed1, self.seed2, self.seed3, self.seed4, self.seed5, self.seed6 = self.random()

    def random(self):
        seed = torch.rand(6) * 0.7
        zs1 = int(torch.clamp(seed[0] * 10, min=0, max=6))
        zs2 = int(torch.clamp(seed[1] * 10, min=0, max=6))
        zs3 = int(torch.clamp(seed[2] * 10, min=0, max=6))
        zs4 = int(torch.clamp(seed[3] * 10, min=0, max=6))
        zs5 = int(torch.clamp(seed[4] * 10, min=0, max=6))
        zs6 = int(torch.clamp(seed[5] * 10, min=0, max=6))
        return zs1, zs2, zs3, zs4, zs5, zs6

    def predict_batch(self, users, items):
        """
        批量预测用户-物品对的评分
        Args:
            users: 用户ID列表 (Tensor)
            items: 物品ID列表 (Tensor)
        Returns:
            scores: 预测分数 (Tensor)
        """
        userInput = self.userMatrix[users, :]  # (B, nItems)
        itemInput = self.itemMatrix[items, :]  # (B, nUsers)

        userVector = self.userFC(userInput)  # (B, fcLayers[0]//2)
        itemVector = self.itemFC(itemInput)  # (B, fcLayers[0]//2)

        y0 = torch.cat((userVector, itemVector), -1)  # (B, fcLayers[0])

        # 通过所有全连接层
        y = self.fcs(y0)

        # 最终预测
        y = self.final(y).squeeze()  # (B,)

        return y

    def recommend(self, user, unrated_items, topk=100, batch_size=1024):
        """
        为指定用户生成推荐列表
        Args:
            user: 用户ID (标量)
            unrated_items: 该用户未交互的物品列表 (Tensor或list)
            topk: 返回的推荐物品数量
            batch_size: 批处理大小，避免显存爆炸
        Returns:
            recommended_items: 推荐的物品ID列表
            scores: 对应的预测分数
        """
        # 确保unrated_items在GPU上
        if isinstance(unrated_items, list):
            unrated_items = torch.tensor(unrated_items, device=self.device)
        elif unrated_items.device != self.device:
            unrated_items = unrated_items.to(self.device)

        n_items = len(unrated_items)
        all_scores = []

        # 分批处理避免显存爆炸
        with torch.no_grad():
            for i in range(0, n_items, batch_size):
                end_idx = min(i + batch_size, n_items)
                batch_items = unrated_items[i:end_idx]

                # 创建重复的用户ID
                batch_users = torch.full((len(batch_items),), user,
                                         device=self.device, dtype=torch.long)

                # 批量预测
                batch_scores = self.predict_batch(batch_users, batch_items).view(-1)
                all_scores.append(batch_scores)

        # 合并所有分数
        if all_scores:
            all_scores = torch.cat(all_scores)
        else:
            all_scores = torch.tensor([], device=self.device)

        # 获取topk推荐
        if len(all_scores) > 0:
            topk_val = min(topk, len(all_scores))
            topk_scores, topk_indices = torch.topk(all_scores, k=topk_val)
            recommended_items = unrated_items[topk_indices].cpu().numpy().tolist()
            recommended_scores = topk_scores.cpu().numpy().tolist()
        else:
            recommended_items = np.array([]).tolist()
            recommended_scores = np.array([]).tolist()

        return recommended_items, recommended_scores
