#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
epsilon = 50  # Learning rate 学习率
momentum = 0.7  # Momentum parameter 动量优化参数
epoch = 1  # Initial epoch 初始化epoch
maxepoch = 50  # Total number of training epochs 总训练次数
err_train = np.zeros(maxepoch)  # Training error 训练误差
err_valid = np.zeros(maxepoch)  # Validation error 验证误差
err_random = np.zeros(maxepoch)  # Random error 随机误差
num_feat = 10  # Number of latent factors 隐因子数量
N = 10000  # Number of training triplets per epoch 每次训练三元组的数量


# In[2]:


train_data = pd.read_csv(
    "data/training.txt", sep=" ", header=None, names=["user_id", "item_id", "click"]
)
train_data["click"] = 5
rating = train_data["click"]
movie = train_data["item_id"]
user = train_data["user_id"]
data_num = len(rating)
movie_num = movie.nunique()  # 电影的数量
user_num = user.nunique()  # 用户的数量


# In[3]:

# 负采样
negative_sample = np.random.choice(user_num * movie_num, data_num, replace=True)
record_user = []
record_movie = []
record_rating = [1] * data_num

for i in negative_sample:
    movie_i = i // user_num
    user_i = i % user_num + 1
    record_user.append(user_i)
    record_movie.append(movie_i)

negative_data = pd.DataFrame(
    {"user_id": record_user, "item_id": record_movie, "click": record_rating}
)


# In[4]:

# 数据结合
data = pd.concat([train_data, negative_data])
rating = data["click"]
movie = data["item_id"]
user = data["user_id"]
data_num = len(rating)
movie_num = movie.max()
user_num = user.max()
data = data.sample(frac=1).reset_index(drop=True)  # 打乱数据集


# In[5]:

# 稀疏度
sparsity = data_num / (movie_num * user_num)
print(f"Sparsity: {sparsity:.6f}")


# In[6]:


train_num = 80000
train_vec = data.iloc[:train_num]
probe_vec = data.iloc[train_num:84000]
mean_rating = train_vec["click"].mean()
pairs_tr = len(train_vec)
pairs_pr = len(probe_vec)
numbatches = 8
num_m = movie_num
num_p = user_num


# In[7]:


# 初始化矩阵参数
w1_M1 = 0.1 * np.random.rand(num_m, num_feat)  # 电影特征矩阵，维度为(num_m, num_feat)
w1_P1 = 0.1 * np.random.rand(num_p, num_feat)  # 用户特征矩阵，维度为(num_p, num_feat)
w1_M1_inc = np.zeros((num_m, num_feat))  # 电影特征矩阵的增量，用于动量优化，初始化为零矩阵
w1_P1_inc = np.zeros((num_p, num_feat))  # 用户特征矩阵的增量，用于动量优化，初始化为零矩阵


# In[8]:


# 训练模型
for epoch in range(maxepoch):
    for batch in range(numbatches):
        start = batch * N
        end = start + N
        if end > pairs_tr:
            end = pairs_tr

        aa_p = train_vec.iloc[start:end]["user_id"].values - 1
        aa_m = train_vec.iloc[start:end]["item_id"].values - 1
        rating = train_vec.iloc[start:end]["click"].values.astype(
            float
        )  # 确保 rating 为浮点数
        rating -= mean_rating

        pred_out = np.sum(w1_M1[aa_m] * w1_P1[aa_p], axis=1)
        f = np.sum((pred_out - rating) ** 2)

        IO = 2 * (pred_out - rating)
        IO = np.tile(IO[:, None], num_feat)

        Ix_m = IO * w1_P1[aa_p]
        Ix_p = IO * w1_M1[aa_m]

        dw1_M1 = np.zeros((movie_num, num_feat))
        dw1_P1 = np.zeros((user_num, num_feat))

        for ii in range(N):
            dw1_M1[aa_m[ii]] += Ix_m[ii]
            dw1_P1[aa_p[ii]] += Ix_p[ii]

        w1_M1_inc = momentum * w1_M1_inc + epsilon * dw1_M1 / N
        w1_M1 -= w1_M1_inc
        w1_P1_inc = momentum * w1_P1_inc + epsilon * dw1_P1 / N
        w1_P1 -= w1_P1_inc

    pred_out = np.sum(w1_M1[aa_m] * w1_P1[aa_p], axis=1)
    f_s = np.sum((pred_out - rating) ** 2)
    err_train[epoch] = np.sqrt(f_s / N)

    aa_p = probe_vec["user_id"].values - 1
    aa_m = probe_vec["item_id"].values - 1
    rating = probe_vec["click"].values

    pred_out = np.sum(w1_M1[aa_m] * w1_P1[aa_p], axis=1) + mean_rating
    pred_out = np.clip(pred_out, 1, 5)

    err_valid[epoch] = np.sqrt(np.sum((pred_out - rating) ** 2) / pairs_pr)

    print(
        f"Epoch {epoch + 1}/{maxepoch}, Train RMSE: {err_train[epoch]:.4f}, Test RMSE: {err_valid[epoch]:.4f}"
    )

# 绘制误差曲线
plt.plot(range(1, maxepoch + 1), err_train, label="Train Error", color="blue")
plt.plot(range(1, maxepoch + 1), err_valid, label="Validation Error", color="red")
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.legend()
plt.show()


# In[9]:


# 定义常量
j = 10


# In[10]:


# 计算Precision@10
precisions = []
for i in range(user_num):
    user_i_rating_real = probe_vec[probe_vec["user_id"] == i + 1]
    user_i_rating_real = user_i_rating_real.sort_values(by="click", ascending=False)
    user_i_rating = (
        np.dot(w1_P1[i, :], w1_M1[user_i_rating_real["item_id"].values - 1].T)
        + mean_rating
    )

    if len(user_i_rating_real) > j:
        top_j_pred = np.argsort(-user_i_rating)[:j]
        precision = np.sum(np.isin(top_j_pred, np.arange(j))) / j
        precisions.append(precision)
    else:
        ti = np.sum(user_i_rating_real["click"] >= 4)
        if ti != 0:
            precision = np.sum(user_i_rating[:ti] >= 4) / ti
            precisions.append(precision)

Pre = np.mean(precisions)


# In[11]:


# 计算Recall@10
recalls = []
for i in range(user_num):
    user_i_rating_real = probe_vec[probe_vec["user_id"] == i + 1]
    user_i_rating_real = user_i_rating_real.sort_values(by="click", ascending=False)
    user_i_rating = (
        np.dot(w1_P1[i, :], w1_M1[user_i_rating_real["item_id"].values - 1].T)
        + mean_rating
    )

    if len(user_i_rating_real) > j:
        user_i_rating[user_i_rating < 4] = 0
        user_i_rating[user_i_rating >= 4] = 1
        ti = np.sum(user_i_rating)
        if ti != 0:
            bigerthan4 = np.sum(user_i_rating_real["click"] >= 4)
            tinri = np.sum(user_i_rating[:bigerthan4])
            recall = tinri / ti
            recalls.append(recall)
    else:
        ti = np.sum(user_i_rating_real["click"] >= 4)
        if ti != 0:
            recall = np.sum(user_i_rating[:ti] >= 4) / ti
            recalls.append(recall)

Re = np.mean(recalls)


# In[12]:


print(f"Recall@10: {Re:.4f}, Precision@10: {Pre:.4f}")


# In[13]:


# 读取测试集数据
test = pd.read_csv("data/test.txt", header=None, sep=" ")
test.columns = ["user_id"]  # 设置列名为'user_id'

# 定义常量
k = 10  # 推荐的数量

# 初始化变量
record = []  # 存储推荐结果的列表

# 对每个测试用户进行推荐
for i in test["user_id"]:
    user_i_rating = np.dot(w1_P1[i - 1, :], w1_M1.T) + mean_rating
    used = data[data["user_id"] == i]["item_id"].values - 1
    user_i_rating[used] = 0  # 将的评分设为0
    top_k_movies = np.argsort(user_i_rating)[-k:][::-1] + 1
    record.append(top_k_movies)  # 将推荐结果加入列表

# 将推荐结果与测试集用户结合
record_df = pd.DataFrame(
    record, columns=[f"recommended_movie_{k}" for k in range(1, k + 1)]
)  # 创建推荐结果的DataFrame
final_result = pd.concat([test, record_df], axis=1)  # 合并测试集用户和推荐结果

# 保存到CSV文件
final_result.to_csv("data/result.csv", index=False)  # 将最终结果保存为CSV文件
