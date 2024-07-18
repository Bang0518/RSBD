import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise.model_selection import cross_validate

data_path = "data/"
# 读取数据
train_data = pd.read_csv(data_path + "training.txt", sep=" ", names=["user_id", "item_id", "click"])
test_data = pd.read_csv(data_path + "test.txt", names=["user_id"])

# 划分数据集
train, val = train_test_split(train_data, test_size=0.2, random_state=42)

# 转换为Surprise库格式
reader = Reader(rating_scale=(0, 1))
train_dataset = Dataset.load_from_df(train[["user_id", "item_id", "click"]], reader)

# 定义一个函数来评估模型
def evaluate_model(algo):
    cross_val_results = cross_validate(algo, train_dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return cross_val_results

# 定义要评估的模型
models = {
    'SVD': SVD(),
    'KNNBasic': KNNBasic(),
    'NMF': NMF()
}

# 评估每个模型并存储结果
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model)

# 绘制性能比较图
def plot_results(results):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for model, result in results.items():
        rmse_scores = result['test_rmse']
        mae_scores = result['test_mae']
        ax[0].plot(rmse_scores, label=model)
        ax[1].plot(mae_scores, label=model)
    ax[0].set_title('RMSE Comparison')
    ax[1].set_title('MAE Comparison')
    ax[0].set_xlabel('Fold')
    ax[1].set_xlabel('Fold')
    ax[0].set_ylabel('RMSE')
    ax[1].set_ylabel('MAE')
    ax[0].legend()
    ax[1].legend()
    plt.show()

plot_results(results)

# 训练最佳模型（以SVD为例）
best_model = SVD()
trainset = train_dataset.build_full_trainset()
best_model.fit(trainset)

# 为测试集中的每个用户生成Top-10推荐
test_user_ids = test_data["user_id"].unique()
recommendations = {}

for user_id in test_user_ids:
    items = train_data["item_id"].unique()
    predictions = [best_model.predict(user_id, item_id) for item_id in items]
    top_10_items = sorted(predictions, key=lambda x: x.est, reverse=True)[:10]
    recommendations[user_id] = [pred.iid for pred in top_10_items]

# 保存结果
with open(data_path + "result.txt", "w") as f:
    for user_id, items in recommendations.items():
        f.write(f"{user_id}: {','.join(map(str, items))}\n")
