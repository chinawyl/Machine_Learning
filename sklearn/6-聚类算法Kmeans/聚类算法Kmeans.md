## 1 概述

### 1.1 无监督学习与聚类算法

决策树，随机森林，逻辑回归，他们虽然有着不同的功能，但却都属于“**有监督学习**”的一部分，即是说，模型在训练的时候，**既需要特征矩阵X，也需要真实标签y**。还有相当一部分算法属于“**无监督学习**”，无监督的算法在训练的时候**只需要特征矩阵X，不需要标签y**。PCA降维算法就是无监督学习中的一种，聚类算法，也是无监督学习的代表算法之一。

聚类算法又叫做“无监督分类”，其目的是将数据划分成有意义或有用的组（或簇）。比如在商业中，如果我们手头有大量的当前和潜在客户的信息，我们可以使用聚类将客户划分为若干组，以便进一步分析和开展营销活动，最有名的客户价值判断模型**RFM**，就常常和聚类分析共同使用。再比如，**聚类可以用于降维和矢量量化**，可以将高维特征压缩到一列当中，常常用于图像，声音，视频等非结构化数据，可以大幅度压缩数据量。

### 1.2聚类和分类区别

|            | 聚类                                                         | 分类                                                         |
| ---------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| 核 心      | 将数据分成多个组探索每个组的数据是否有联系                   | 从已经分组的数据中去学习把新数据放到已经分好的组中去         |
| 学 习 类型 | 无监督，无需标签进行训练                                     | 有监督，需要标签进行训练                                     |
| 典型算法   | K-Means，DBSCAN，层次聚类，光谱聚类                          | 决策树，贝叶斯，逻辑回归                                     |
| 算法输出   | 聚类结果是不确定的，不一定总是能够反映数据的真实分类，同样的聚类，根据不同的业务需求，可能是一个好结果，也可能是一个坏结果 | 分类结果是确定的，分类的优劣是客观的，不是根据业务或算法需求决定 |

<br>

## 2 KMeans

### 2.1 KMeans是如何工作的

##### 2.1.1 簇与质心

**簇:**KMeans算法将一组N个样本的特征矩阵X划分为K个无交集的簇，直观上来看是簇是一组一组聚集在一起的数
据，在一个簇中的数据就认为是同一类。簇就是聚类的结果表现。

**质心:**簇中所有数据的均值 通常被称为这个簇的“质心”（centroids）。在一个二维平面中，一簇数据点的质心的
横坐标就是这一簇数据点的横坐标的均值，质心的纵坐标就是这一簇数据点的纵坐标的均值。同理可推广至高
维空间。

##### 2.1.2 寻找簇与质心顺序过程

| 顺序 | 过程                                                         |
| ---- | ------------------------------------------------------------ |
| 1    | 随机抽取K个样本作为最初的质心                                |
| 2    | 开始循环                                                     |
| 2.1  | 将每个样本点分配到离他们最近的质心，生成K个簇                |
| 2.2  | 对于每个簇，计算所有被分到该簇的样本点的平均值作为新的质心   |
| 3    | 当质心的位置不再发生变化，迭代停止，聚类完成1 随机抽取K个样本作为最初的质心 |

### 2.2 簇内误差平方和的定义和解惑

#### 2.2.1 平方和的定义

**被分在同一个簇中的数据是有相似性的，而不同簇中的数据是不同的**，这与“分箱”概念有些类似，即我们分箱的目的是希望，一个箱内的人有着相似的信用风险，而不同箱的人的信用风险差异巨大，以此来区别不同信用度的人，因此我们追求“组内差异小，组间差异大”。聚类算法也是同样的目的，我们追求“**簇内差异小，簇外差异大**”。而这个“差异“，由**样本点到其所在簇的质心的距离来衡量**。

对于一个簇来说，所有样本点到质心的距离之和越小，我们就认为这个簇中的样本越相似，簇内差异就越小。而距
离的衡量方法有多种，令x表示簇中的一个样本点，μ表示该簇中的质心，n表示每个样本点中的特征数目，i表示组成点x的每个特征，则该样本点到质心的距离可以由以下距离来度量：
$$
欧几里得距离：d(x,\mu)=\sqrt{\sum_{i=1}^{n}(x_i-\mu_i)^2}
$$

$$
曼哈顿距离：d(x,\mu)=\sum_{i=1}^{n}(|x_i-\mu|)
$$

$$
余弦距离：\cos\theta=\frac{\sum_1^n(x_i*\mu)}{\sum_1^n(x_i)^2*\sum_1^n(\mu)^2}
$$

如我们采用欧几里得距离，则一个簇中所有样本点到质心的距离的平方和为：
$$
Cluster\ Sum\ of\ Square\ (CSS)=\sum_{j=0}^{m}\sum_{i=1}^n(x_i-\mu_i)^2
$$

$$
Total\ Cluster\ Sum\ of\ Square=\sum_{l=1}^kCSS_l
$$

其中，m为一个簇中样本的个数，j是每个样本的编号。这个公式被称为**簇内平方和**（cluster Sum of Square），
又叫做Inertia。而将一个数据集中的所有簇的簇内平方和相加，就得到了整体平方和（Total Cluster Sum of
Square），又叫做total inertia。Total Inertia越小，代表着每个簇内样本越相似，聚类的效果就越好。因此，
**KMeans追求的是，求解能够让Inertia最小化的质心**。实际上，在质心不断变化不断迭代的过程中，总体平方和
是越来越小的。我们可以使用数学来证明，当整体平方和最小的时候，质心就不再发生变化了。如此，K-Means的求解过程，就变成了一个最优化问题。

实际上，我们也可以使用其他距离，每个距离都有自己对应的Inertia。在过去的经验中，我们总结出不同距离所对应的质心选择方法和Inertia，在Kmeans中，只要使用了正确的质心和距离组合，无论使用什么样的距离，都可以达到不错的聚类效果：

| 距离度量     | 质心   | Inertia                                |
| ------------ | ------ | -------------------------------------- |
| 欧几里得距离 | 均值   | 最小化每个样本点到质心的欧式距离之和   |
| 曼哈顿距离   | 中位数 | 最小化每个样本点到质心的曼哈顿距离之和 |
| 余弦距离     | 均值   | 最小化每个样本点到质心的余弦距离之和   |

#### 2.2.2 解惑(Kmeans是否有损失函数)

在逻辑回归中曾有这样的结论：损失函数本质是用来衡量模型的拟合效果的，只有有着求解参数需求的算法，才会有损失函数。**Kmeans不求解什么参数，它的模型本质也没有在拟合数据**，而是在对数据进行一种探索。所以如果你去问大多数数据挖掘工程师，甚至是算法工程师，他们可能会告诉你说，K-Means不存在什么损失函数,Inertia更像是Kmeans的模型评估指标，而非损失函数。

但我们类比过了Kmeans中的Inertia和逻辑回归中的损失函数的功能，我们发现它们确实非常相似。所以，从“求解模型中的某种信息，用于后续模型的使用“这样的功能来看，我们可以认为Inertia是Kmeans中的损失函数，虽然这种说法并不严谨。

对比来看，在决策树中，我们有衡量分类效果的指标准确度accuracy，准确度所对应的损失叫做泛化误差，但
**我们不能通过最小化泛化误差来求解某个模型中需要的信息**，**我们只是希望模型的效果上表现出来的泛化误差**
**很小**。因此决策树，KNN等算法，是绝对没有损失函数的。

### 2.3 KMeans算法的时间复杂度

KMeans算法的平均复杂度是O(k*n*T)，其中k是我们的超参数，所需要输入的簇数，n是整个数据集中的样本量，T是所需要的迭代次数（相对的，KNN的平均复杂度是O(n)）。在最坏的情况下，KMeans的复杂度可以写作，其中n是整个数据集中的样本量，p是特征总数。这个最高复杂度是由D. Arthur和S. Vassilvitskii在2006年发表的文”k-means方法有多慢？“中提出的。
在实践中，比起其他聚类算法，k-means算法已经快了，但它一般找到Inertia的局部最小值。 这就是为什么多次
重启它会很有用。

<br>

## 3 sklearn.cluster.KMeans

### 3.1 重要参数n_clusters

n_clusters是KMeans中的k，表示着我们告诉模型我们要分几类。这是KMeans当中唯一一个必填的参数，**默认为8类**，但通常我们的聚类结果会是一个小于8的结果。通常，在开始聚类之前，我们并不知道n_clusters究竟是多少

#### 3.1.1 先进行一次聚类

```python
# 1.导入库和模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 2.创建数据集
X, y = make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)

# 3.未分簇时
fig, ax1 = plt.subplots(1)
ax1.scatter(X[:, 0], X[:, 1]
           ,marker='o' #点的形状
           ,s=8 #点的大小
           )
plt.show()

# 4.正确分簇时
color = ["red","pink","orange","gray"]
fig, ax1 = plt.subplots(1)
for i in range(4):
    ax1.scatter(X[y==i, 0], X[y==i, 1]
           ,marker='o' #点的形状
           ,s=8 #点的大小
           ,c=color[i]
           )
plt.show()

# 5.重要参数n_clusters（基本概念）

n_clusters = 3

#分簇
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X) #不需要调用接口，训练时就已经找到质心分好簇了
y_pred = cluster.labels_ #查看聚合好的类别，每个样本所对应的类
print(y_pred)
print()
pre = cluster.fit_predict(X) #与调用labels_效果一样
print(pre == y_pred)
print()

#数据量大的时候使用局部数据和predict，但只是相似
# cluster_smallsub = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:200])
# y_pred_ = cluster_smallsub.predict(X)
# y_pred == y_pred_

#查看质心
centroid = cluster.cluster_centers_
print(centroid)
print()

#查看总距离平方和
inertia = cluster.inertia_
print(inertia)

#绘图
color = ["red","pink","orange","gray"]
fig, ax1 = plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(X[y_pred==i, 0], X[y_pred==i, 1]
           ,marker='o'
           ,s=8
           ,c=color[i]
           )
ax1.scatter(centroid[:,0],centroid[:,1]
           ,marker="x"
           ,s=15
           ,c="black")
plt.show()

# 6.重要参数n_clusters（查看inertia）
#KMeans不能因为调节n_clusters使得inertia降低而说模型变好，要保持n_clusters不变
n_clusters = 4
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
print(inertia_)
n_clusters = 5
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
print(inertia_)
n_clusters = 6
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
print(inertia_)

'''
908.3855684760613
811.0952123653019
728.2271936260031

'''
```

#### 3.1.2 聚类算法的模型评估指标

KMeans的目标是确保“簇内差异小，簇外差异大”，我们就可以通过衡量簇内差异来衡量聚类的效果。Inertia是用距离来衡量簇内差异的指标，因此，我们是否可以使用Inertia来作为聚类的衡量指标呢？可以，但是这个指标的缺点和极限太大。
**第一：**它不是有界的。我们只知道，Inertia是越小越好，是0最好，但我们不知道，一个较小的Inertia究竟有没有
达到模型的极限，能否继续提高。
**第二：**它的计算太容易受到特征数目的影响，数据维度很大的时候，Inertia的计算量会陷入维度诅咒之中，计算量会爆炸，不适合用来一次次评估模型。
**第三：**它会受到超参数K的影响，在我们之前的常识中其实我们已经发现，随着K越大，Inertia注定会越来越小，但这并不代表模型的效果越来越好了
**第四：**Inertia对数据的分布有假设，它假设数据满足凸分布（即数据在二维平面图像上看起来是一个凸函数的样
子），并且它假设数据是各向同性的（isotropic），即是说数据的属性在不同方向上代表着相同的含义。但是现实中的数据往往不是这样。所以使用Inertia作为评估指标，会让聚类算法在一些细长簇，环形簇，或者不规则形状的流形时表现不佳：

![001-聚类算法的模型评估指标](D:\Machine_Learning\sklearn\6-聚类算法Kmeans\images\001-聚类算法的模型评估指标.png)

那我们可以使用什么指标呢？分两种情况来看。

##### 3.1.2.1 当真实标签已知的时候

虽然我们在聚类中不输入真实标签，但这不代表我们拥有的数据中一定不具有真实标签，或者一定没有任何参考信
息。当然，在现实中，拥有真实标签的情况非常少见（几乎是不可能的）。如果拥有真实标签，我们更倾向于使用
分类算法。但不排除我们依然可能使用聚类算法的可能性。如果我们有样本真实聚类情况的数据，我们可以对于聚
类算法的结果和真实结果来衡量聚类的效果。常用的有以下三种方法：

| 模型评估指标                                                 | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| **互信息分**<br/>普通互信息分<br/>metrics.adjusted_mutual_info_score (y_pred, y_true)<br/>调整的互信息分<br/>metrics.mutual_info_score (y_pred, y_true)<br/>标准化互信息分<br/>metrics.normalized_mutual_info_score (y_pred, y_true)<br/>取值范围在(0,1)之中 | 取值范围在(0,1)之中<br/>越接近1，聚类效果越好<br/>在随机均匀聚类下产生0分 |
| **V-measure**：基于条件上分析的一系列直观度量<br/>同质性：是否每个簇仅包含单个类的样本<br/>metrics.homogeneity_score(y_true, y_pred)<br/>完整性：是否给定类的所有样本都被分配给同一个簇中<br/>metrics.completeness_score(y_true, y_pred)<br/>同质性和完整性的调和平均，叫做V-measure<br/>metrics.v_measure_score(labels_true, labels_pred)<br/>三者可以被一次性计算出来：<br/>metrics.homogeneity_completeness_v_measure(labels_true,<br/>labels_pred) | 取值范围在(0,1)之中<br/>越接近1，聚类效果越好<br/>由于分为同质性和完整性两种度量，可以更仔细地研究，模型到底哪个任务<br/>做得不够好<br/>对样本分布没有假设，在任何分布上都可以有不错的表现<br/>在随机均匀聚类下不会产生0分 |
| **调整兰德系数**<br/>metrics.adjusted_rand_score(y_true, y_pred) | 取值在(-1,1)之间，负值象征着簇内的点差异巨大，甚至相互独立，正类的<br/>兰德系数比较优秀，越接近1越好<br/>对样本分布没有假设，在任何分布上都可以有不错的表现，尤其是在具<br/>有"折叠"形状的数据上表现优秀<br/>在随机均匀聚类下产生0分 |

##### 3.1.2.2 当真实标签未知的时候：轮廓系数

在99%的情况下，我们是对没有真实标签的数据进行探索，也就是对不知道真正答案的数据进行聚类。这样的聚
类，是完全依赖于评价簇内的稠密程度（簇内差异小）和簇间的离散程度（簇外差异大）来评估聚类的效果。其中
轮廓系数是最常用的聚类算法的评价指标。它是对每个样本来定义的，它能够同时衡量：
1）样本与其自身所在的簇中的其他样本的相似度**a**，等于样本与同一簇中所有其他点之间的平均距离
2）样本与其他簇中的样本的相似度**b**，等于样本与下一个最近的簇中的所有点之间的平均距离

根据聚类的要求”**簇内差异小，簇外差异大**“，我们希望**b永远大于a**，并且大得越多越好。
单个样本的轮廓系数计算为：
$$
s = \frac{b-a}{max(a,b)}
$$
这个公式可以被解析为：
$$
s=
	\begin{cases}
		1-a/b,&\text{if $a<b$}\\
		0,&\text{if $a=b$}\\
		b/a-1,&\text{if $a>b$}
	\end{cases}
$$
很容易理解**轮廓系数范围是(-1,1)**，其中值越接近1表示样本与自己所在的簇中的样本很相似，并且与其他簇中的样本不相似，当样本点与簇外的样本更相似的时候，轮廓系数就为负。当轮廓系数为0时，则代表两个簇中的样本相似度一致，两个簇本应该是一个簇。可以总结为**轮廓系数越接近于1越好，负数则表示聚类效果非常差**。

如果一个簇中的大多数样本具有比较高的轮廓系数，则簇会有较高的总轮廓系数，则整个数据集的平均轮廓系数越
高，则聚类是合适的。如果**许多样本点具有低轮廓系数甚至负值，则聚类是不合适的，聚类的超参数K可能设定得**
**太大或者太小**。

在sklearn中，我们使用模块metrics中的类**silhouette_score**来计算轮廓系数，它返回的是一个数据集中，所有样本的轮廓系数的均值。但我们还有同在metrics模块中的**silhouette_sample**，它的参数与轮廓系数一致，但返回的是数据集中每个样本自己的轮廓系数。

```python
# 1.导入库和模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

# 2.创建数据集
X, y = make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)

# 3.返回一个数据集中，所有样本的轮廓系数的均值
for i in range(3,7):
    n_clusters = i #分簇数量
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X) #不需要调用接口，训练时就已经找到质心分好簇了
    y_pred = cluster.labels_ #查看聚合好的类别，每个样本所对应的类
    re = silhouette_score(X,y_pred)
    print(str(i)+'  '+str(re))
    
# 4.返回数据集中每个样本自己的轮廓系数
n_clusters = 4 #分簇数量
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X) #不需要调用接口，训练时就已经找到质心分好簇了
y_pred = cluster.labels_ #查看聚合好的类别，每个样本所对应的类
silhouette_samples(X,y_pred)
```

##### 3.1.2.3 当真实标签未知的时候：Calinski-Harabaz Index

| 标签未知时的评估指标                                         |
| ------------------------------------------------------------ |
| 卡林斯基-哈拉巴斯指数<br/>sklearn.metrics.calinski_harabaz_score (X, y_pred) |
| 戴维斯-布尔丁指数<br/>sklearn.metrics.davies_bouldin_score (X, y_pred) |
| 权变矩阵<br/>sklearn.metrics.cluster.contingency_matrix (X, y_pred) |

们重点来了解一下卡林斯基-哈拉巴斯指数。Calinski-Harabaz**指数越高越好**。对于有k个簇的聚类而言，
Calinski-Harabaz指数s(k)写作如下公式：
$$
s(k)=\frac{Tr(B_k)}{Tr(W_k)}*\frac{N-k}{k-1}
$$
其中N为数据集中的样本量，k为簇的个数（即类别的个数），Bk是组间离散矩阵，即不同簇之间的协方差矩阵，
Wk是簇内离散矩阵，即一个簇内数据的协方差矩阵，而tr表示矩阵的迹。在线性代数中，一个n×n矩阵A的主对角
线（从左上方至右下方的对角线）上各个元素的总和被称为矩阵A的迹（或迹数），一般记作tr(A) 。数据之间的离
散程度越高，协方差矩阵的迹就会越大。组内离散程度低，协方差的迹就会越小，Tr(Wk)也就越小，同时，组间
离散程度大，协方差的的迹也会越大，Tr(Bk)就越大，这正是我们希望的，因此Calinski-harabaz指数越高越好。

**比较轮廓系数与哈拉巴斯指数运行时间**

```python
# 5.比较轮廓系数与哈拉巴斯指数运行时间
from sklearn.metrics import calinski_harabaz_score
from time import time

n_clusters = 4 #分簇数量
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X) #不需要调用接口，训练时就已经找到质心分好簇了
y_pred = cluster.labels_ #查看聚合好的类别，每个样本所对应的类

#哈拉巴斯指数运行时间
t0 = time()
calinski_harabaz_score(X, y_pred)
print(time() - t0)

#轮廓系数运行时间
t0 = time()
silhouette_score(X,y_pred)
print(time() - t0)

'''
0.0009734630584716797
0.005513906478881836
可以看得出，calinski-harabaz指数比轮廓系数的计算块了一倍不止。想想看我们使用的数据量，如果是一个以万
计的数据，轮廓系数就会大大拖慢我们模型的运行速度了。
'''
```

#### 3.1.3 基于轮廓系数来选择n_clusters

```python
# 1.导入库和模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# 2.创建数据集并实例化
X, y = make_blobs(n_samples=500,n_features=2,centers=4,random_state=1)

# 3.绘制图片
for n_clusters in [2,3,4,5,6,7]:
    
    #设置画布
    n_clusters = n_clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1]) #设置x轴
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10]) #设置y轴
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
    
    #求出轮廓系数与哈拉巴斯指数
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10 #是图像不靠近x轴
    
    #对每一个簇进行循环
    for i in range(n_clusters):
        #从每个样本的轮廓系数结果中抽取出第i个簇的轮廓系数并进行排序
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        #查看一个簇中有多少样本
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i #y轴取值为y初始值加该簇样本数取值
        color = cm.nipy_spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper)
                         ,ith_cluster_silhouette_values
                         ,facecolor=color
                         ,alpha=0.7
                         )
        ax1.text(-0.05
                 , y_lower + 0.5 * size_cluster_i
                 , str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1]
               ,marker='o'
               ,s=8
               ,c=colors
               )
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='x',
                c="red", alpha=1, s=200)
    
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()
```

### 3.2 重要参数init & random_state & n_init：初始质心怎么放好

在K-Means中有一个重要的环节，就是放置初始质心。如果有足够的时间，K-means一定会收敛，但Inertia可能收敛到局部最小值。是否能够收敛到真正的最小值很大程度上取决于质心的初始化。**init**就是用来帮助我们**决定初始化方式**的参数。

在之前讲解初始质心的放置时，我们是使用”随机“的方法在样本点中抽取k个样本作为初始质心，这种方法显然不符合”稳定且更快“的需求。为此，我们可以使用**random_state**参数来**控制每次生成的初始质心都在相同位置**，甚至可以画学习曲线来确定最优的random_state是哪个整数。

一个random_state对应一个质心随机初始化的随机数种子。如果不指定随机数种子，则sklearn中的K-means并不
会只选择一个随机模式扔出结果，而会在每个随机数种子下运行多次，并使用结果最好的一个随机数种子来作为初
始质心。我们可以使用参数**n_init**来选择，**每个随机数种子下运行的次数**。这个参数不常用到，**默认10次**，如果我
们希望运行的结果更加精确，那我们可以增加这个参数n_init的值来增加每个随机数种子下运行的次数。

##### 3.2.1 init

可以输入"k-means++"，"random"或者一个n维数组。这是初始化质心的方法，**默认为"k-means++"**。通常都是输入"kmeans++"：一种为K均值聚类选择初始聚类中心的聪明的办法，以加速收敛。如果输入了**n维数组**，数组的形状应该是**(n_clusters，n_features)**并给出初始质心。

##### 3.2.1 random_state

控制每次质心随机初始化的随机数种子

##### 3.2.2 n_init

整数，默认10，使用不同的质心随机初始化的种子来运行k-means算法的次数。最终结果会是基于Inertia
来计算的n_init次连续运行后的最佳输出

### 3.3 重要参数max_iter & tol：让迭代停下来

在之前描述K-Means的基本流程时我们提到过，当质心不再移动，Kmeans算法就会停下来。但在完全收敛之前，
我们也可以使用max_iter，最大迭代次数，或者tol，两次迭代间Inertia下降的量，这两个参数来让迭代提前停下
来。有时候，当我们的n_clusters选择不符合数据的自然分布，或者我们为了业务需求，必须要填入与数据的自然
分布不合的n_clusters，提前让迭代停下来反而能够提升模型的表现。

##### 3.3.1 max_iter

整数，默认300，单次运行的k-means算法的最大迭代次数

##### 3.3.2 tol

浮点数，默认1e-4，两次迭代间Inertia下降的量，如果两次迭代之间Inertia下降的值小于tol所设定的值，迭
代就会停下

```python
random = KMeans(n_clusters = 10,init="random",max_iter=10,random_state=420).fit(X)
y_pred_max10 = random.labels_
silhouette_score(X,y_pred_max10)

random = KMeans(n_clusters = 10,init="random",max_iter=20,random_state=420).fit(X)
y_pred_max20 = random.labels_
silhouette_score(X,y_pred_max20)
```

### 3.4 重要属性与重要接口

![002-重要属性与重要接口](D:\Machine_Learning\sklearn\6-聚类算法Kmeans\images\002-重要属性与重要接口.png)

### 3.5 函数cluster.k_means

```python
from sklearn.cluster import k_means
k_means(X,4,return_n_iter=True) #return_n_iter默认False，不显示迭代次数
```

## 4 案例：聚类算法用于降维，KMeans的矢量量化应用

```python
# 1.导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin #对两个序列中的点进行举例匹配
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle #打乱排序

# 2.导入数据，探索数据
china = load_sample_image("china.jpg")
print(china)
print()
print(china.dtype)
print()
print(china.shape) #长度*宽度*像素
print()
print(china[0][0] ) #三个数决定一种颜色
print()

newimage = china.reshape((427 * 640,3)) #改变维度
print(pd.DataFrame(newimage).drop_duplicates().shape) #去除重复值
plt.figure(figsize=(15,15))
plt.imshow(china) #必须导入三维数组形成的图片

# 3.决定超参数，数据预处理
n_clusters = 64 #降到64种颜色

#plt.imshow在浮点数上表现更优秀
china = np.array(china, dtype=np.float64) / china.max() #数据归一化

#将图像格式转换成矩阵格式
w, h, d = original_shape = tuple(china.shape) #保存长度，宽度，像素
assert d == 3 #assert判断是否为True，不满足就报错
image_array = np.reshape(china, (w * h, d)) #改变维度为2维
print(image_array)
print()
print(image_array.shape)

# 4. 对数据进行K-Means的矢量量化
#首先使用1000个数据找出质心
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array_sample)
print(kmeans.cluster_centers_.shape) #1000个样本返回64个质心
print()

#按照已存在的质心对所有数据进行聚类
labels = kmeans.predict(image_array)
print(labels.shape)
print()

#使用质心替换掉所有样本
image_kmeans = image_array.copy() #包含去重后的9万多种去重的颜色
for i in range(w*h):
    image_kmeans[i] = kmeans.cluster_centers_[labels[i]]
print(pd.DataFrame(image_kmeans).drop_duplicates().shape)
print()
image_kmeans = image_kmeans.reshape(w,h,d)
print(image_kmeans.shape)

# 5.对数据进行随机的矢量量化
centroid_random = shuffle(image_array, random_state=0)[:n_clusters]
labels_random = pairwise_distances_argmin(centroid_random,image_array,axis=0)
print(labels_random.shape)
print()
len(set(labels_random))
image_random = image_array.copy()
for i in range(w*h):
    image_random[i] = centroid_random[labels_random[i]]
image_random = image_random.reshape(w,h,d)
print(image_random.shape)

# 6.将原图，按KMeans矢量量化和随机矢量量化的图像绘制出来
plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(image_kmeans)
plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(image_random)
plt.show()
```

