import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv("data/training.txt", header=None, sep=' ')
data.columns = ['user_id', 'item_id', 'rating']
user_num = data['user_id'].nunique()
item_num = data['item_id'].nunique()

# 初始化评分矩阵
rating_matrix = np.zeros((user_num, item_num))
for row in data.itertuples():
    rating_matrix[row.user_id - 1, row.item_id - 1] = row.rating

# 计算每个用户的平均评分
mean_rating = np.mean(rating_matrix, axis=1)
mean_rating[np.isnan(mean_rating)] = 0

# 减去平均评分
normalized_rating_matrix = rating_matrix - mean_rating[:, np.newaxis]

# SVD分解
U, sigma, Vt = np.linalg.svd(normalized_rating_matrix, full_matrices=False)
sigma = np.diag(sigma)

# 保留前k个奇异值
k = 50
w1_P1 = np.dot(U[:, :k], sigma[:k, :k])
w1_M1 = Vt[:k, :]

# 计算推荐评分
recalls = []
precisions = []
for i in range(user_num):
    user_i_rating = np.dot(w1_P1[i, :], w1_M1) + mean_rating[i]
    user_i_rating_real = data[data['user_id'] == i + 1]
    if not user_i_rating_real.empty:
        user_i_rating[user_i_rating < 4] = 0
        user_i_rating[user_i_rating >= 4] = 1
        ti = np.sum(user_i_rating)
        if ti != 0:
            bigerthan4 = np.sum(user_i_rating_real['click'] >= 4)
            tinri = np.sum(user_i_rating[:bigerthan4])
            recall = tinri / ti
            recalls.append(recall)
            precision = tinri / j  # 计算 Precision@10
            precisions.append(precision)
    else:
        ti = np.sum(user_i_rating_real['click'] >= 4)
        if ti != 0:
            recall = np.sum(user_i_rating[:ti] >= 4) / ti
            recalls.append(recall)
            precision = np.sum(user_i_rating[:ti] >= 4) / j  # 计算 Precision@10
            precisions.append(precision)

Re = np.mean(recalls)
Pre = np.mean(precisions)

print(f'Recall@10: {Re:.4f}, Precision@10: {Pre:.4f}')

# 读取测试集数据
test = pd.read_csv("data/test.txt", header=None, sep=' ')
test.columns = ['user_id']

# 定义常量
j = 10

# 初始化变量
record = []

# 对每个测试用户进行推荐
for i in test['user_id']:
    user_i_rating = np.dot(w1_P1[i - 1, :], w1_M1.T) + mean_rating
    used = data[data['user_id'] == i]['item_id'].values - 1  # 减1以匹配索引
    user_i_rating[used] = 0
    top_j_movies = np.argsort(user_i_rating)[-j:][::-1] + 1  # 加1以匹配实际的电影ID
    record.append(top_j_movies)

# 将推荐结果与测试集用户结合
record_df = pd.DataFrame(record, columns=[f'recommended_movie_{k}' for k in range(1, j + 1)])
final_result = pd.concat([test, record_df], axis=1)

# 保存到CSV文件
final_result.to_csv('result.csv', index=False)
