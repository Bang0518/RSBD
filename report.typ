#import "template.typ": *

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

training.txt是用户-物品-隐式反馈的交互对，一共有四万多条交互信息。在代码中将其拆分为训练集和验证集。
test.txt是真实的测试集，只有用户ID，我们最终需要在该测试集上进行Top-N推荐任务。
result.txt是算法得到的结果，即对test.txt中的用户一一进行Top-10推荐。
== 实验设计 <setting>

本实验包括以下几个步骤。
== 实验环境 <environment>

本实验使用的环境如下：
- Ubuntu 22.04 LTS
- Python 3.11.9
- Packages in `requirement.txt`

= 实现细节 <details>

== 数据加载与预处理

这部分导入了必要的库并设置全局参数，包括字体设置和超参数（学习率、动量参数、最大训练次数等）。

```Python

```
== 数据集划分和稀疏度计算

== 初始化矩阵参数

== 训练模型

== 计算Precision@10和Recall@10

== 推荐结果生成与保存

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