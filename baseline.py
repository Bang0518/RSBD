import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import coo_matrix

data_path = "data/"
# 读取数据
train_data = pd.read_csv(data_path + "training.txt", sep=" ", names=["user_id", "item_id", "click"])
test_data = pd.read_csv(data_path + "test.txt", names=["user_id"])

# 映射用户ID和物品ID为连续的整数
user_id_map = {id: index for index, id in enumerate(train_data["user_id"].unique())}
item_id_map = {id: index for index, id in enumerate(train_data["item_id"].unique())}

train_data["user_id"] = train_data["user_id"].map(user_id_map)
train_data["item_id"] = train_data["item_id"].map(item_id_map)
test_data["user_id"] = test_data["user_id"].map(user_id_map)

# 负采样
def negative_sampling(df, num_samples):
    users = df["user_id"].values
    items = df["item_id"].values
    negative_samples = []
    for _ in range(num_samples):
        user = np.random.choice(users)
        item = np.random.choice(items)
        while (df[(df["user_id"] == user) & (df["item_id"] == item)].shape[0] > 0):
            user = np.random.choice(users)
            item = np.random.choice(items)
        negative_samples.append([user, item, 1])
    return pd.DataFrame(negative_samples, columns=["user_id", "item_id", "click"])

neg_samples = negative_sampling(train_data, len(train_data))
train_data["click"] = 5
train_data = pd.concat([train_data, neg_samples])

# 划分数据集
train, val = train_test_split(train_data, test_size=0.2, random_state=42)

# 构建稀疏矩阵
def build_sparse_matrix(df):
    return coo_matrix((df["click"], (df["user_id"], df["item_id"])), shape=(len(user_id_map), len(item_id_map)))

train_sparse = build_sparse_matrix(train)
val_sparse = build_sparse_matrix(val)

# 定义评估函数
def evaluate_model(model, val_sparse, K=10):
    user_count, item_count = val_sparse.shape
    precision_at_k = []
    recall_at_k = []
    for user in range(user_count):
        val_items = val_sparse.tocsr()[user].indices
        if len(val_items) == 0:
            continue
        recommended_items = model.recommend(user, val_sparse.tocsr(), N=K, filter_already_liked_items=False)[0].tolist()
        hits = len(set(recommended_items) & set(val_items))
        precision_at_k.append(hits / K)
        recall_at_k.append(hits / len(val_items))
    
    return np.mean(precision_at_k), np.mean(recall_at_k)

# 训练模型并评估
models = {
    'ALS': AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50),
    'BPR': BayesianPersonalizedRanking(factors=20, regularization=0.1, iterations=50)
}

results = {}
for name, model in models.items():
    model.fit(train_sparse.T)
    results[name] = evaluate_model(model, val_sparse)

# 绘制性能比较图
def plot_results(results):
    plt.rcParams["font.family"] = "Times New Roman"
    
    precisions = {name: result[0] for name, result in results.items()}
    recalls = {name: result[1] for name, result in results.items()}
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    ax[0].bar(precisions.keys(), precisions.values(), color='skyblue')
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('Precision@K')
    ax[0].set_title('Precision at K Comparison')
    
    ax[1].bar(recalls.keys(), recalls.values(), color='lightgreen')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('Recall@K')
    ax[1].set_title('Recall at K Comparison')
    
    plt.show()

plot_results(results)