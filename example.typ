#import "simplepaper.typ": *

#show: project.with(
  header_text: "2024 暑期大数据推荐系统课程",
  title: "基于隐式反馈的 Top-10 推荐列表预测",
  authors: (
    (
      name: "吴建军",
      organization: [学号2024140899],
      email: "roshibang@bupt.edu.cn"
    ),

  ),
  abstract: "本实验通过使用提供的训练集构建了推荐模型，并使用测试集中的数据对推荐系统进行了Top-10推荐列表的预测。完成了数据处理、模型构建、训练与验证的过程，并生成了每个用户的推荐结果。通过实验验证了推荐算法的效果和可行性，并对推荐系统的性能进行了评估。",
  keywords: (
    "推荐系统",
    "Top-10推荐",
    "隐式反馈",
  ),
)

= 实验概述 <introduction>

利用训练集 training.txt 内的信息对测试集内的用户进行 top-10 预测 (即，为每一个测试集内用户，从商品集合中预测 10 个最有可能被点击的商品。) 。提示: 可以首先自行将给定的训练集进一步划分，如取 80%训练集数据为新的训练集和 20%数据为验证集，然后以新的训练集训练不同的推荐算法模型，并在验证集上验证推荐效果，最终选择在验证集上最优的推荐算法对测试集内用户进行 top-10 推荐预测。
== 实验设计 <setting>

本实验包括以下几个步骤。
=== 数据预处理

从文件中读取训练数据（training.txt）和测试数据（test.txt）。将训练数据集划分为训练集和验证集，比例为80%和20%。
=== 模型训练与评估

本实验选用了三种推荐算法模型：SVD模型、KNNBasic模型和NMF模型。
- SVD模型（Singular Value Decomposition）。SVD是一种矩阵分解技术，用于降维和处理高维数据。在推荐系统中，SVD将用户-物品评分矩阵分解为三个矩阵的乘积，从而预测用户对未评分物品的评分。本实验使用交叉验证（Cross Validation）技术，对模型进行5折交叉验证，评估指标为RMSE（均方根误差）和MAE（平均绝对误差）。
- KNNBasic模型（K-Nearest Neighbors Basic）。KNN是一种基于实例的学习算法，用于分类和回归。在推荐系统中，KNNBasic可以基于用户或物品的相似性进行评分预测。同样使用5折交叉验证评估模型的RMSE和MAE。
- NMF模型（Non-negative Matrix Factorization）。NMF是一种非负矩阵分解技术，用于将用户-物品评分矩阵分解为两个非负矩阵的乘积。在推荐系统中，NMF能够捕捉用户和物品的隐含特征。使用5折交叉验证评估模型的RMSE和MAE。
=== 性能比较

将各模型的交叉验证结果进行比较，使用Matplotlib绘制RMSE和MAE的比较图表，以直观展示各模型的性能差异。
=== 推荐生成

选择在验证集上表现最优的模型对其进行全量训练，并生成测试集用户的Top-10推荐列表。
=== 结果保存

将生成的推荐结果保存到文件中，文件格式为user_id: item1_id,item2_id,...,item10_id。
== 实验环境 <environment>

本实验使用的环境如下：
- Python 3.11.9
- 

= 实现细节 <details>

使用前确保已经安装了对应的字体！详细字体列表参考 #link("https://github.com/1bitbool/SimplePaper/tree/main")[README] 或模板文件。

= 使用示例 <example>

== 特殊标记 <bug1>

你可以 Typst 的语法对文本进行特殊标记，模板设定了几种语法的样式：*突出*、_强调_、引用@example。


== 图片

图片标题默认设置了方正楷体，效果如@img1 如果你想要使用其他字体，你可以自行修改模版。

#figure(image("images/nju-emblem.svg"),
  caption: [
    示例图片
  ],
)<img1>

图片后的文字。

== 表格

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
    inset: 10pt,
    align: horizon,
    [], [周一], [周二],[周三],[周四],[周五],
    "早上", "编译原理", "操作系统", "计算机网络", "操作系统", "计算机网络",
    "下午", "数据挖掘", "计算机网络", "操作系统", "计算机网络", "分布式系统"
  ),
  caption: "示例表格"
)

表格后的文字。

== 代码

我们为代码添加了如下简单的样式。

```c
#include <stdio.h>
int main()
{
   // printf() 中字符串需要引号
   printf("Hello, World!");
   return 0;
}
```

代码后的文字。

== 列表

下面是有序列表的示例：

+ 第一项
+ 第二项
+ 第三项

列表后的文字。

下面是无序列表的示例：

- 第一项
- 第二项
- 第三项

无序列表后的文字。

#bibliography("ref.bib")