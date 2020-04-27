## 1 概述

### 1.1 什么叫“维度”

对于**数组**和**Series**来说，**维度就是功能shape返回的结果，shape中返回了几个数字，就是几维**。索引以外的数
据，不分行列的叫一维（此时shape返回唯一的维度上的数据个数），有行列之分叫二维（shape返回行x列），也称为表。一张表最多二维，复数的表构成了更高的维度。当一个数组中存在2张3行4列的表时，shape返回的是(更高维，行，列)。当数组中存在2组2张3行4列的表时，数据就是4维，shape返回(2,2,3,4)。

![001-数组Series](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\001-数组Series.png)

数组中的每一张表，都可以是一个**特征矩阵**或一个**DataFrame**，这些结构永远只有一张表，所以一定有行列，其中行是样本，列是特征。针对每一张表，**维度指的是样本的数量或特征的数量，一般无特别说明，指的都是特征的数量**。除了索引之外，一个特征是一维，两个特征是二维，n个特征是n维。

![002-Dataframe特征矩阵](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\002-Dataframe特征矩阵.png)

**对图像来说，维度就是图像中特征向量的数量**。特征向量可以理解为是坐标轴，一个特征向量定义一条直线，是一维，两个相互垂直的特征向量定义一个平面，即一个直角坐标系，就是二维，三个相互垂直的特征向量定义一个空
间，即一个立体直角坐标系，就是三维。三个以上的特征向量相互垂直，定义人眼无法看见，也无法想象的高维空
间。![003-图像](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\003-图像.png)



**降维算法中的”降维“，指的是降低特征矩阵中特征的数量**。，降维的目的是为了让**算法运算更快，效果更好**，但其实还有另一种需求：**数据可视化**。从上面的图我们其实可以看得出，图像和特征矩阵的维度是可以相互对应的，即一个特征对应一个特征向量，对应一条坐标轴。所以，三维及以下的特征矩阵，是可以被可视化的，这可以帮助我们很快地理解数据的分布，而三维以上特征矩阵的则不能被可视化，数据的性质也就比较难理
解。

<br>

### 1.2 sklearn中的降维算法

sklearn中降维算法都被包括在模块decomposition中，这个模块本质是一个矩阵分解模块。在过去的十年中，如
果要讨论算法进步的先锋，矩阵分解可以说是独树一帜。矩阵分解可以用在降维，深度学习，聚类分析，数据预处
理，低纬度特征学习，推荐系统，大数据分析等领域。在2006年，Netflix曾经举办了一个奖金为100万美元的推荐系统算法比赛，最后的获奖者就使用了矩阵分解中的：奇异值分解SVD

![004-降维算法](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\004-sklearn降维算法.png)

<br>

## 2 PCA与SVD

**方差过滤:**如果一个特征的方差很小，则意味着这个特征上很可能有大量取值都相同（比如90%都是1，只有10%是0，甚至100%是1），那这一个特征的取值对样本而言就没有区分度，这种特征就不带有有效信息。从方差的这种应用就可以推断出，**如果一个特征的方差很大，则说明这个特征上带有大量的信息**。因此，在降维中，**PCA使用的信息量衡量指标，就是样本方差，又称可解释性方差，方差越大，特征所带的信息量越多**。
$$
Var = \frac{1}{n-1}\sum_{i=1}^n(x_i-\bar x)^2
$$
Var代表一个特征的方差，n代表样本量，xi代表一个特征中的每个样本取值，xhat代表这一列样本的均值。

### 2.1 降维实现

![005-降维实现](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\005-降维实现.png)

#### 2.1.1 原始坐标系方差

我们现在有一组简单的数据，有特征x1和x2，三个样本数据的坐标点分别为(1,1)，(2,2)，(3,3)。我们可以让x1和
x2分别作为两个特征向量，很轻松地用一个二维平面来描述这组数据。这组数据现在每个特征的均值都为2，**方差**
**则等于:**
$$
x1_var = x2_var = \frac{(1-2)^2+(2-2)^2+(3-2)^2}{3-1} = 1
$$
**每个特征的数据一模一样，因此方差也都为1，数据的方差总和是2**

#### 2.1.2 变换后坐标系方差

现在我们的目标是：只用一个特征向量来描述这组数据，即将二维数据降为一维数据，并且尽可能地保留信息量，
即让数据的**总方差**尽量靠近2。于是，我们将原本的直角坐标系逆时针旋转45°，形成了新的特征向量x1和x2组
成的新平面，在这个新平面中，三个样本数据的坐标点发生了更改，可以注意到，x2上的数值此时都变成了0，因此x2明显不带有任何有效信息了（此时x2的方差也为0了）。此时**方差可表示成:**
$$
x1_var = \frac{(\sqrt2-2\sqrt2)^2+(2\sqrt2-2\sqrt2)^2+(3\sqrt2-2\sqrt2)^2}{3-1} = 2
$$
**x2上的数据均值为0，方差也为0**

#### 注:

##### 1.方差计算公式中除数之所以是n-1，是为了得到样本方差的无偏估计

##### 2.PCA和SVD是两种不同的降维算法，但他们都遵从上面的过程来实现降维，只是两种算法中矩阵分解的方法不同，信息量的衡量指标不同罢了。

##### 3.PCA和SVD数学原理

PCA使用**方差**作为信息量的衡量指标，并且特征值分解来找出空间V。降维时，它会通过一系列数学的神秘操作（比如说，产生协方差矩阵**1/n乘X乘X^T**）将特征矩阵X分解为以下三个矩阵，其中 **Q和Q^-1是辅助的矩阵**，**Σ是一个对角矩阵**（即除了对角线上有值，其他位置都是0的矩阵），其对角线上的元素就是方差。降维完成之后，PCA找到的每个新特征向量就叫做“主成分”，而被丢弃的特征向量被认为信息量很少，这些信息很可能就是噪音。

![006-PCA数学宇宙](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\006-PCA数学宇宙.png)

SVD使用**奇异值**分解来找出空间V，其中Σ也是一个对角矩阵，不过它对角线上的元素是奇异值。这也是SVD中用
来衡量特征上的信息量的指标。**U和V^{T}分别是左奇异矩阵和右奇异矩阵**，也都是辅助矩阵

![007-SVD数学宇宙](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\007-SVD数学宇宙.png)

<br>

### 2.2 重要参数n_components

n_components是我们降维后需要的维度，即降维后需要保留的特征数量，降维流程中第二步里需要确认的k值，
一般输入**[0, min(X.shape)]**范围中的整数。如果我们希望可视化一组数据来观察数据分布，我们往往将数据降到**三维以下**，很多时候是二维，即n_components的取值为2。

#### 2.2.1 简单案例：高维数据的可视化

```python
# 1.导入库和模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 2.提取数据集
iris = load_iris() 
X = iris.data
y = iris.target

# 3.建模

#实例化
pca = PCA(n_components=2)

#拟合模型
pca = pca.fit(X)

#获取新矩阵
X_matrix = pca.transform(X)

# 4.可视化
colors = ['red','yellow','blue']
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_matrix[y == i,0]
                ,X_matrix[y == i,1]
                ,alpha=.7
                ,c=colors[i]
                ,label=iris.target_names[i]
               )
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()

# 5.探索降维后的数据

#查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_)
print()

#查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比（又叫做可解释方差贡献率）
print(pca.explained_variance_ratio_) #大部分信息都被有效地集中在了第一个特征上
print()

#查看降维后的新特征矩阵的信息在原始矩阵的信息占比
print(pca.explained_variance_ratio_.sum())

# 6.选择最好的n_components：累积可解释方差贡献率曲线
pca_line = PCA().fit(X)
plt.plot([1,2,3,4],np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1,2,3,4]) #这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()
```

#### 2.2.2 最大似然估计自选超参数

```python
# 最大似然估计自选超参数
pca_mle = PCA(n_components="mle")
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)

#mle为我们自动选择了3个特征
print(X_mle)
print()

#得到了比设定2个特征时更高的信息含量
print(pca_mle.explained_variance_ratio_.sum())
```

#### 2.2.3 按信息量占比选超参数

```python
# 按信息量占比选超参数
pca_f = PCA(n_components=0.97,svd_solver="full")
pca_f = pca_f.fit(X)
X_f = pca_f.transform(X)
pca_f.explained_variance_ratio_
```

<br>

### 2.3 PCA中的SVD

#### 2.3.1 PCA中的SVD哪里来？

SVD有一种惊人的数学性质，即是它可以**跳过数学神秘的宇宙，不计算协方差矩阵，直接找出一个新特征向量组成的n维空间**，而这个n维空间就是奇异值分解后的右矩阵**V^T**（所以一开始在讲解降维过程时，我们说”生成新特征向量组成的空间V"，并非巧合，而是特指奇异值分解中的矩阵 **V^T**）。

![008-SVD开挂](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\008-SVD开挂.png)

k就是n_components，是我们降维后希望得到的维度。若X为(m,n)的特征矩阵， **V^T** 就是结构为(n,n)的矩阵，取这个矩阵的前k行（进行切片），即将V转换为结构为(k,n)的矩阵。而$V_{(k,n)}^T$与原特征矩阵X相乘，即可得到降
维后的特征矩阵X_dr。这是说，**奇异值分解可以不计算协方差矩阵等等结构复杂计算冗长的矩阵，就直接求出新特征空间和降维后的特征矩阵**

<br>

##### sklearn实现过程

![009-skleran具体过程](D:\Machine_Learning\sklearn\主成分分析PCA与奇异值分解SVD\images\009-skleran具体过程.png)

在sklearn中，矩阵U和Σ虽然会被计算出来（同样也是一种比起PCA来说简化非常多的数学过程，不产生
协方差矩阵），但完全不会被用到，也无法调取查看或者使用，因此我们可以认为，**U和Σ在fit过后就被遗弃了**。奇异值分解追求的仅仅是V，只要有了V，就可以计算出降维后的特征矩阵。在transform过程之后，**fit中奇异值分解的结果除了V(k,n)以外，就会被舍弃**，而V(k,n)会被保存在属性components_ 当中，可以调用查看

```python
PCA(2).fit(X).components_

PCA(2).fit(X).components_.shape
```

<br>

#### 2.3.2 重要参数svd_solver 与 random_state

**1.svd_solver:**有四种模式可选,**默认”auto"**。

​	**(1).auto：**基于X.shape和n_components的默认策略来选择分解器：如果**输入数据的尺寸大于500x500**且要提
​	取的**特征数小于数据最小维度min(X.shape)的80％**，就启用效率更高的”randomized“方法。否则，精确完整
​	的SVD将被计算，截断将会在矩阵被分解完成后有选择地发生

​	**(2).full：**从scipy.linalg.svd中调用标准的LAPACK分解器来生成精确完整的SVD，适合数据量比较适中，计算时
​	间充足的情况，生成的精确完整的SVD的结构为：	
$$
U_{(m,n)},\sum(m,n),V^T_{(n,n)}
$$
​	**(3).arpack：**从scipy.sparse.linalg.svds调用ARPACK分解器来运行截断奇异值分解(SVD truncated)，分解时就
​	将特征数量降到n_components中输入的数值k，**可以加快运算速度，适合特征矩阵很大的时候，但一般用于特**	**征矩阵为稀疏矩阵的情况**，此过程包含一定的随机性。截断后的SVD分解出的结构为：
$$
U_{(m,k)},\sum(k,k),V^T_{(n,n)}
$$
​	**(4).randomized：**通过Halko等人的随机方法进行随机SVD。在"full"方法中，分解器会根据原始数据和输入的
​	n_components值去计算和寻找符合需求的新特征向量，但是在"randomized"方法中，分解器会先生成多个
​	随机向量，然后一一去检测这些随机向量中是否有任何一个符合我们的分解需求，如果符合，就保留这个随
​	机向量，并基于这个随机向量来构建后续的向量空间。这个方法已经被Halko等人证明，比"full"模式下计算快
​	很多，并且还能够保证模型运行效果。**适合特征矩阵巨大，计算量庞大的情况**。

**2.random_state:**在参数svd_solver的值为"arpack" or "randomized"的时候生效，可以控制这两种SVD模式中
的随机模式。通常我们就选用”auto“，不必对这个参数纠结太多。

<br>

