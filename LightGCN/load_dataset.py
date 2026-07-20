import numpy as np
import random
import scipy as sp
import torch
import pickle
import json
import os
from tqdm import tqdm
from scipy.sparse import dok_matrix
from scipy.sparse import vstack


# class Load():

#     def __init__(self, train_rating_path, fake_users_file=None):
#         self.train_rating = self.load_rating(train_rating_path)
#         self.train_group = self.get_train_group()
#         self.num_users = int(torch.max(self.train_rating[:, 0])) + 1
#         self.num_items = int(torch.max(self.train_rating[:, 1])) + 1

#         # 加载训练矩阵，确保列数足够
#         self.trainMatrix = self.load_rating_file_as_matrix(train_rating_path, min_cols=self.num_items)
#         # 记录原始用户数量（在添加假用户之前）
#         self.original_num_users = self.num_users
#         if fake_users_file is not None:
#             self.poison_path = fake_users_file
#             self._load_fakes()
#             self.target_users, self.unrated_items_dict = self.get_target_users_and_unrated_items()
#         else:
#             self.target_items = set()
#             self.target_users, self.unrated_items_dict = None, None

#     def load_rating(self, path):
#         rating = np.loadtxt(path, delimiter='\t')
#         record_count = len(rating[:, 0])
#         user_count = int(max(rating[:, 0])) + 1
#         item_count = int(max(rating[:, 1])) + 1
#         # item_count = 12929  # AM-usic

        
#         print('Loaded:', path)
#         print('Num of users:', user_count)
#         print('Num of items:', item_count)
#         print('Data sparsity:', record_count / (user_count * item_count))
#         # remove the last column: timestamp
#         return torch.from_numpy(rating[:, :-1])#movielens -2
    
#     def load_rating_file_as_matrix(self, filename, min_cols=None):
#         '''
#         Read .rating file and Return dok matrix.
#         The first line of .rating file is: num_users\t num_items
#         '''
#         # Get number of users and items
#         num_users, num_items = 0, 0
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line is not None and line != "":
#                 arr = line.split("\t")
#                 u, i = int(arr[0]), int(arr[1])
#                 num_users = max(num_users, u)
#                 num_items = max(num_items, i)
#                 line = f.readline()

#         # 如果指定了最小列数，确保矩阵足够大
#         if min_cols is not None and min_cols > num_items:
#             num_items = min_cols - 1  # -1 因为后面会+1

#         # Construct matrix
#         mat = dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
#         with open(filename, "r") as f:
#             line = f.readline()
#             while line is not None and line != "":
#                 arr = line.split("\t")
#                 user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
#                 if rating > 0:
#                     mat[user, item] = 1.0
#                 line = f.readline()
#         return mat
#     def _load_fakes(self):
#         """加载假用户数据并合并到训练矩阵，同时记录目标项目"""
#         self.target_items = set()  # 初始化目标项目集合
#         fake_file_path = self.poison_path
#         if not os.path.exists(fake_file_path):
#             return

#         with open(fake_file_path, 'r') as f:
#             fake_data = json.load(f)
#         fake_user_interactions, target_items = fake_data["fake_users"], fake_data["target_items"]
#         self.target_items = set(target_items)  # 确保是集合

#         # 检查假用户数据中的最大物品ID
#         max_fake_item = 0
#         for _, fake_interaction in fake_user_interactions.items():
#             if fake_interaction:
#                 max_fake_item = max(max_fake_item, max(fake_interaction))

#         # 如果假用户数据中有更大的物品ID，扩展训练矩阵
#         if max_fake_item >= self.num_items:
#             self._extend_train_matrix(max_fake_item + 1)

#         new_rows = []
#         for _, fake_interaction in fake_user_interactions.items():
#             if len(fake_interaction) < 10:
#                 continue
#             new_row = dok_matrix((1, self.num_items), dtype=np.float32)
#             for item in fake_interaction:
#                 new_row[0, item] = 1.0
#             new_rows.append(new_row)

#         if new_rows:
#             new_user_matrix = vstack(new_rows)
#             self.trainMatrix = vstack([self.trainMatrix, new_user_matrix]).todok()
#             self.num_users = self.trainMatrix.shape[0]

#     def get_target_users_and_unrated_items(self):
#         """
#         寻找普通用户中与目标物品没有交集的用户，并返回这些用户及其未交互物品

#         Returns:
#             target_users: 列表，与目标物品没有交集的普通用户ID
#             unrated_items_dict: 字典，键为用户ID，值为该用户未交互的物品列表
#         """
#         target_users = []
#         unrated_items_dict = {}


#         # 遍历所有原始普通用户（不包括后来添加的假用户）
#         for user_id in range(self.original_num_users):
#             # 获取该用户交互过的物品
#             user_interactions = self.trainMatrix[user_id]
#             rated_items = set(user_interactions.nonzero()[1])  # 获取用户交互过的物品索引

#             # 检查用户是否与任何目标物品有交互
#             has_target_interaction = any(item in rated_items for item in self.target_items)

#             # 如果用户没有与任何目标物品交互，则将其加入目标用户列表
#             if not has_target_interaction:
#                 target_users.append(user_id)

#                 # 获取该用户的所有未交互物品
#                 all_items = set(range(self.num_items))
#                 unrated_items = list(all_items - rated_items)
#                 unrated_items_dict[user_id] = unrated_items

#         return target_users, unrated_items_dict


#     def get_train_group(self):
#         neg = {}

#         if os.path.exists('./Data/train_neg_dict.json'):
#             neg = self.load_neg_dict_from_json('./Data/train_neg_dict.json')            
#         else:
#             neg = self.get_negative(self.train_rating, 1000)#构建负样本池
#             self.save_neg_dict_to_json(neg, './Data/train_neg_dict.json')

#         # save negative sample for resampling
#         self.train_negative = neg
        
#         record_count = len(self.train_rating[:, 0])
#         groups = []
#         for r in range(record_count):
#             u = int(self.train_rating[r, 0])
#             i = int(self.train_rating[r, 1])
#             j = int(random.sample(neg[u], 1)[0])#从负样本池中选一个
#             groups.append([u, i, j])
#         return torch.tensor(groups)

        
#     def get_negative(self, data, sample_count):
#         print('Calculating negative samples...')
#         neg = {}
#         record_count = len(data[:, 0])
#         user_count = int(max(data[:, 0])) + 1
#         item_count = int(max(data[:, 1])) + 1
#         for u in range(user_count):
#             neg[u] = []
#         last_u = 0
#         neg[0] = set(range(item_count))
#         # record_count = 100
#         for r in tqdm(range(record_count)):
#             u = int(data[r, 0])
#             if u != last_u:
#                 neg[last_u] = random.sample(list(neg[last_u]), sample_count)
#                 neg[u] = set(range(item_count))
#             last_u = u
#             i = int(data[r, 1])
#             neg[u] = neg[u] - set([i])
#         # neg[last_u] = set(range(item_count))
#         neg[last_u] = random.sample(list(neg[last_u]), sample_count)

#         return neg



#     def save_neg_dict_to_json(self, neg, path):
#         print('Saving negative samples to file:', path)
#         with open(path, "w") as f:
#             json.dump(neg, f, sort_keys=True)

#     def load_neg_dict_from_json(self, path):
#         print('Loading negative samples from file', path)
#         with open(path, 'r') as f:
#             d = json.load(f)
#             keys = list(map(int, d.keys()))
#             neg = {}
#             for i in range(len(keys)):
#                neg[keys[i]] = list(d.values())[i]
#             return neg


class Load():

    def __init__(self, train_rating_path, fake_file_file=None, attack_type=None):
        self.train_rating_path = train_rating_path + '.train.rating'
        self.train_rating = self.load_rating(self.train_rating_path)
        self.attack_type = attack_type
        # 记录初始 user/item 数，用于区分真实用户 vs 假用户
        self.original_num_users = int(torch.max(self.train_rating[:, 0]).item()) + 1
        self.num_items = int(torch.max(self.train_rating[:, 1]).item()) + 1
        # === 初始化 trainMatrix 用于扩展（稀疏矩阵更好，但此处直接 List 转 Tensor即可）===
        self.trainMatrix = self.train_rating.clone()
        # === 加载假用户数据（如有）===
        if fake_file_file is not None:
            self.fake_file_path = fake_file_file
            self._load_fakes()
        else:
            self.target_items = set()

        # === 构建最终三元组 BPR 训练集 ===
        self.train_group = self.get_train_group()



    def load_rating(self, path):
        rating = np.loadtxt(path, delimiter='\t')
        record_count = len(rating[:, 0])
        user_count = int(max(rating[:, 0])) + 1
        item_count = int(max(rating[:, 1])) + 1
        # item_count = 12929  # AM-usic

        
        print('Loaded:', path)
        print('Num of users:', user_count)
        print('Num of items:', item_count)
        print('Data sparsity:', record_count / (user_count * item_count))
        # remove the last column: timestamp
        return torch.from_numpy(rating[:, :-1])#movielens -2

    def _load_fakes(self):
        """读取 fake 用户，追加到 trainMatrix"""
        if not os.path.exists(self.fake_file_path):
            return
        
        with open(self.fake_file_path, "r") as f:
            fake_data = json.load(f)

        fake_users = fake_data["fake_users"]
        target_items = fake_data["target_items"]
        self.target_items = set(target_items)

        # 可能含有更大 itemID，需要扩展 item 数
        non_empty_item_lists = [v for v in fake_users.values() if len(v) > 0]
        if non_empty_item_lists:
            max_fake_item = max(max(v) for v in non_empty_item_lists)
            if max_fake_item >= self.num_items:
                self.num_items = max_fake_item + 1
        # 同时检查target_items是否需要扩展item数
        elif target_items:
            max_target_item = max(target_items)
            if max_target_item >= self.num_items:
                self.num_items = max_target_item + 1

        # 追加 fake users -> trainMatrix
        appended = []
        next_user_id = self.original_num_users
        train_cols = self.trainMatrix.size(1)   # 动态检测列数

        for _, item_list in fake_users.items():
            if len(item_list) == 0:
                continue
            for itm in item_list:

                if train_cols == 2:
                    # 数据格式为 [u, i]
                    appended.append([next_user_id, itm])

                elif train_cols == 3:
                    # 数据格式为 [u, i, rating/timestamp]
                    appended.append([next_user_id, itm, 1])  # rating 填1即可

                else:
                    raise ValueError(f"Unexpected trainMatrix.shape[1] = {train_cols}")
            next_user_id += 1

        if len(appended) > 0:
            appended = torch.tensor(appended).long()
            self.trainMatrix = torch.cat([self.trainMatrix, appended], dim=0)

        self.num_users = next_user_id
        print(f"Added Fake users. Total users = {self.num_users}, items = {self.num_items}")

    def get_train_group(self):
        neg = {}

        if os.path.exists(self.train_rating_path + f'.{self.attack_type}.train_neg_dict.json'):
            neg = self.load_neg_dict_from_json(self.train_rating_path + f'.{self.attack_type}.train_neg_dict.json')            
        else:
            neg = self.get_negative(self.trainMatrix, 1000)#构建负样本池
            self.save_neg_dict_to_json(neg, self.train_rating_path + f'.{self.attack_type}.train_neg_dict.json')

        # save negative sample for resampling
        self.train_negative = neg
        
        groups = []
        rating = self.trainMatrix
        record_count = len(rating[:, 0])

        for r in range(record_count):
            u = int(rating[r, 0])
            i = int(rating[r, 1])
            j = int(random.sample(neg[u], 1)[0])
            groups.append([u, i, j])

        print(f"[TRAIN GROUP] total triplets = {len(groups)}")
        return torch.tensor(groups).long()

        
    def get_negative(self, data, sample_count):
        print('Calculating negative samples...')
        neg = {}
        record_count = len(data[:, 0])
        user_count = int(max(data[:, 0])) + 1
        item_count = int(max(data[:, 1])) + 1
        for u in range(user_count):
            neg[u] = []
        last_u = 0
        neg[0] = set(range(item_count))
        # record_count = 100
        for r in tqdm(range(record_count)):
            u = int(data[r, 0])
            if u != last_u:
                neg[last_u] = random.sample(list(neg[last_u]), sample_count)
                neg[u] = set(range(item_count))
            last_u = u
            i = int(data[r, 1])
            neg[u] = neg[u] - set([i])
        # neg[last_u] = set(range(item_count))
        neg[last_u] = random.sample(list(neg[last_u]), sample_count)

        return neg



    def save_neg_dict_to_json(self, neg, path):
        print('Saving negative samples to file:', path)
        with open(path, "w") as f:
            json.dump(neg, f, sort_keys=True)

    def load_neg_dict_from_json(self, path):
        print('Loading negative samples from file', path)
        with open(path, 'r') as f:
            d = json.load(f)
            keys = list(map(int, d.keys()))
            neg = {}
            for i in range(len(keys)):
               neg[keys[i]] = list(d.values())[i]
            return neg