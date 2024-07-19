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
    hit_rate = []
    ndcg_at_k = []
    for user in range(user_count):
        val_items = val_sparse.tocsr()[user].indices
        if len(val_items) == 0:
            continue
        recommended_items = model.recommend(user, val_sparse.tocsr(), N=K, filter_already_liked_items=False)[0]
        hits = len(set(recommended_items) & set(val_items))
        precision_at_k.append(hits / K)
        hit_rate.append(hits / min(len(val_items), K))
        
        # 计算NDCG
        rel_scores = [1 if item in val_items else 0 for item in recommended_items]
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(val_items), K))])
        dcg = sum([rel / np.log2(idx + 2) for idx, rel in enumerate(rel_scores)])
        ndcg_at_k.append(dcg / idcg)
    
    return np.mean(precision_at_k), np.mean(hit_rate), np.mean(ndcg_at_k)

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
    precisions = {name: result[0] for name, result in results.items()}
    hit_rates = {name: result[1] for name, result in results.items()}
    ndcgs = {name: result[2] for name, result in results.items()}
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].bar(precisions.keys(), precisions.values(), color='skyblue')
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('Precision@K')
    ax[0].set_title('Precision at K Comparison')
    
    ax[1].bar(hit_rates.keys(), hit_rates.values(), color='lightgreen')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('HR@K')
    ax[1].set_title('Hit Rate at K Comparison')
    
    ax[2].bar(ndcgs.keys(), ndcgs.values(), color='salmon')
    ax[2].set_xlabel('Model')
    ax[2].set_ylabel('NDCG@K')
    ax[2].set_title('NDCG at K Comparison')
    
    plt.show()

plot_results(results)

# 选择最佳模型
best_model_name = max(results, key=lambda x: results[x][0])
best_model = models[best_model_name]
print(f"Best model: {best_model_name}")

# 使用最佳模型生成推荐
best_model.fit(train_sparse.T)
test_user_ids = test_data["user_id"].unique()
recommendations = {}

for user_id in test_user_ids:
    recommended_items = best_model.recommend(user_id, train_sparse.T, N=10, filter_already_liked_items=False)[0]
    original_user_id = list(user_id_map.keys())[list(user_id_map.values()).index(user_id)]
    original_item_ids = [list(item_id_map.keys())[list(item_id_map.values()).index(item)] for item in recommended_items]
    recommendations[original_user_id] = original_item_ids

# 保存结果
with open(data_path + "result.txt", "w") as f:
    for user_id, items in recommendations.items():
        f.write(f"{user_id}: {','.join(map(str, items))}\n")
