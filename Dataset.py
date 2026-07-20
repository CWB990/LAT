import json
import os
import scipy.sparse as sp
import numpy as np
from scipy.sparse import dok_matrix
class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, fake_users_file=None):
        '''
        Constructor
        '''
        # 先加载测试数据来获取最大物品ID
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        # 计算测试数据中的最大物品ID
        max_test_item = self._get_max_item_id()

        # 加载训练矩阵，确保列数足够
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating", min_cols=max_test_item + 1)
        self.num_users, self.num_items = self.trainMatrix.shape

        # 记录原始用户数量（在添加假用户之前）
        self.original_num_users = self.num_users

        if fake_users_file is not None:
            self.poison_path = fake_users_file
            self._load_fakes()
            self.target_users, self.unrated_items_dict = self.get_target_users_and_unrated_items()
        else:
            self.target_items = set()
            self.target_users, self.unrated_items_dict = None, None

    def _get_max_item_id(self):
        """从测试数据中获取最大物品ID"""
        max_item = 0

        # 从testRatings中找最大物品ID
        for user, item, rating in self.testRatings:
            max_item = max(max_item, item)

        # 从testNegatives中找最大物品ID
        for negatives in self.testNegatives:
            for item in negatives:
                max_item = max(max_item, item)

        return max_item

    def _load_fakes(self):
        """加载假用户数据并合并到训练矩阵，同时记录目标项目"""
        self.target_items = set()  # 初始化目标项目集合
        fake_file_path = self.poison_path
        if not os.path.exists(fake_file_path):
            return

        with open(fake_file_path, 'r') as f:
            fake_data = json.load(f)
        fake_user_interactions, target_items = fake_data["fake_users"], fake_data["target_items"]
        self.target_items = set(target_items)  # 确保是集合

        # 检查假用户数据中的最大物品ID
        max_fake_item = 0
        for _, fake_interaction in fake_user_interactions.items():
            if fake_interaction:
                max_fake_item = max(max_fake_item, max(fake_interaction))

        # 如果假用户数据中有更大的物品ID，扩展训练矩阵
        if max_fake_item >= self.num_items:
            self._extend_train_matrix(max_fake_item + 1)

        new_rows = []
        for _, fake_interaction in fake_user_interactions.items():
            if len(fake_interaction) < 10:
                continue
            new_row = sp.dok_matrix((1, self.num_items), dtype=np.float32)
            for item in fake_interaction:
                new_row[0, item] = 1.0
            new_rows.append(new_row)

        if new_rows:
            new_user_matrix = sp.vstack(new_rows)
            self.trainMatrix = sp.vstack([self.trainMatrix, new_user_matrix]).todok()
            self.num_users = self.trainMatrix.shape[0]

    def _extend_train_matrix(self, new_num_items):
        """扩展训练矩阵的列数"""
        if new_num_items <= self.num_items:
            return

        print(f"扩展训练矩阵的列数: {self.num_items} -> {new_num_items}")
        extended_matrix = dok_matrix((self.num_users, new_num_items), dtype=np.float32)

        # 复制原有数据
        for (i, j), value in self.trainMatrix.items():
            extended_matrix[i, j] = value

        self.trainMatrix = extended_matrix
        self.num_items = new_num_items

    def get_target_users_and_unrated_items(self):
        """
        寻找普通用户中与目标物品没有交集的用户，并返回这些用户及其未交互物品

        Returns:
            target_users: 列表，与目标物品没有交集的普通用户ID
            unrated_items_dict: 字典，键为用户ID，值为该用户未交互的物品列表
        """
        target_users = []
        unrated_items_dict = {}


        # 遍历所有原始普通用户（不包括后来添加的假用户）
        for user_id in range(self.original_num_users):
            # 获取该用户交互过的物品
            user_interactions = self.trainMatrix[user_id]
            rated_items = set(user_interactions.nonzero()[1])  # 获取用户交互过的物品索引

            # 检查用户是否与任何目标物品有交互
            has_target_interaction = any(item in rated_items for item in self.target_items)

            # 如果用户没有与任何目标物品交互，则将其加入目标用户列表
            if not has_target_interaction:
                target_users.append(user_id)

                # 获取该用户的所有未交互物品
                all_items = set(range(self.num_items))
                unrated_items = list(all_items - rated_items)
                unrated_items_dict[user_id] = unrated_items

        return target_users, unrated_items_dict

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                ratingList.append([user, item, rating])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:100]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename, min_cols=None):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()

        # 如果指定了最小列数，确保矩阵足够大
        if min_cols is not None and min_cols > num_items:
            num_items = min_cols - 1  # -1 因为后面会+1

        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()
        return mat