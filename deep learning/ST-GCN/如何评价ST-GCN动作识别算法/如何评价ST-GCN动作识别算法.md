# 如何评价ST-GCN动作识别算法？

[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition[paperwithcode]](https://arxiv.org/abs/1801.07455)

作者的主要工作就两点：
- 使用 OpenPose 处理了视频，提出了一个数据集
- 结合 GCN 和 TCN 提出了模型，在数据集上效果还不错

## OpenPose 预处理
&emsp;&emsp;OpenPose 是一个标注人体的关节（颈部，肩膀，肘部等），连接成骨骼，进而估计人体姿态的算法。作为视频的预处理工具，我们只需要关注 OpenPose 的输出就可以了。

![视频中的骨骼标注](image\如何评价ST-GCN动作识别算法\视频中的骨骼标注.html)

&emsp;&emsp;总的来说，视频的骨骼标注结果维数比较高。在一个视频中，可能有很多帧（Frame）。每个帧中，可能存在很多人（Man）。每个人又有很多关节（Joint）。每一个关节又有不同特征（位置、置信度）。

![关节的特征](image\如何评价ST-GCN动作识别算法\关节的特征.html)

&emsp;&emsp;对于一个 batch 的视频，我们可以用一个 5 维矩阵 (N,C,T,V,M) 表示。

- N 代表视频的数量，通常一个 batch 有 256 个视频（其实随便设置，最好是 2 的指数）。

- C 代表关节的特征，通常一个关节包含 x,y,acc等 3 个特征（如果是三维骨骼就是 4 个）。

- T 代表关键帧的数量，一般一个视频有 150 帧。

- V 代表关节的数量，通常一个人标注 18 个关节。

- M 代表一帧中的人数，一般选择平均置信度最高的 2 个人。

&emsp;&emsp;所以，OpenPose 的输出，也就是 ST-GCN 的输入，形状为 (256,3,150,18,2)。

&emsp;&emsp;想要搞 End2End 的同学还是要稍微关注一下 OpenPose 的实现的。最近还有基于 heatmap 的工作，效果也不错~

## ST-GCN 网络结构
&emsp;&emsp;论文中给出的模型描述很丰满，要是只看骨架，网络结构如下：

![ST-GCN 网络结构](image\如何评价ST-GCN动作识别算法\ST-GCN网络结构.html)

主要分为三部分：

![归一化](image\如何评价ST-GCN动作识别算法\关节的特征.html)

&emsp;&emsp;首先，对输入矩阵进行归一化，具体实现如下：

```
N, C, T, V, M = x.size()
# 进行维度交换后记得调用 contiguous 再调用 view 保持显存连续
x = x.permute(0, 4, 3, 1, 2).contiguous()
x = x.view(N * M, V * C, T)
x = self.data_bn(x)
x = x.view(N, M, V, C, T)
x = x.permute(0, 1, 3, 4, 2).contiguous()
x = x.view(N * M, C, T, V)
```

&emsp;&emsp;归一化是在时间和空间维度下进行的（ $V\times C$ ）。也就是将一个关节在不同帧下的位置特征（x 和 y 和 acc）进行归一化。这个操作是利远大于弊的：
- 关节在不同帧下的关节位置变化很大，如果不进行归一化不利于算法收敛
- 在不同 batch 不同帧下的关节位置基本上服从随机分布，不会造成不同 batch 归一化结果相差太大，而导致准确率波动。

![时空变换](image\如何评价ST-GCN动作识别算法\时空变换.html)

&emsp;&emsp;接着，通过 ST-GCN 单元，交替的使用 GCN 和 TCN，对时间和空间维度进行变换：
```
# N*M(256*2)/C(3)/T(150)/V(18)
Input：[512, 3, 150, 18]
ST-GCN-1：[512, 64, 150, 18]
ST-GCN-2：[512, 64, 150, 18]
ST-GCN-3：[512, 64, 150, 18]
ST-GCN-4：[512, 64, 150, 18]
ST-GCN-5：[512, 128, 75, 18]
ST-GCN-6：[512, 128, 75, 18]
ST-GCN-7：[512, 128, 75, 18]
ST-GCN-8：[512, 256, 38, 18]
ST-GCN-9：[512, 256, 38, 18]
```
&emsp;&emsp;空间维度是关节的特征（开始为 3），时间的维度是关键帧数（开始为 150）。在经过所有 ST-GCN 单元的时空卷积后，关节的特征维度增加到 256，关键帧维度降低到 38。

&emsp;&emsp;个人感觉这样设计是因为，人的动作阶段并不多，但是每个阶段内的动作比较复杂。比如，一个挥高尔夫球杆的动作可能只需要分解为 5 步，但是每一步的手部、腰部和脚部动作要求却比较多。

![read out 输出](image\如何评价ST-GCN动作识别算法\readout输出.html)

&emsp;&emsp;最后，使用平均池化、全连接层（或者叫 FCN）对特征进行分类，具体实现如下：
```
# self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

# global pooling
x = F.avg_pool2d(x, x.size()[2:])
x = x.view(N, M, -1, 1, 1).mean(dim=1)
# prediction
x = self.fcn(x)
x = x.view(x.size(0), -1)
```

&emsp;&emsp;Graph 上的平均池化可以理解为对 Graph 进行 read out，即汇总节点特征表示整个 graph 特征的过程。这里的 read out 就是汇总关节特征表示动作特征的过程了。通常我们会使用基于统计的方法，例如对节点求 max,sum,mean 等等。mean 鲁棒性比较好，所以这里使用了 mean。

&emsp;&emsp;插句题外话，这里的$1\times1$ 卷积和全连接层等效，最近在用 matconvnet 的时候，发现它甚至不提供全连接层，只使用 $1\times1$ 的卷积。

## GCN
&emsp;&emsp;从结果上看，最简单的图卷积似乎已经能取得很好的效果了，具体实现如下：
```
def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD
```
作者在实际项目中使用的图卷积公式就是：
$$aggre(x)=D^{-1}AX$$

公式可以进行如下化简：
$$\begin{aligned}
aggre(x)&=D^{-1}AX\\
        &=\sum^N_{k=1}D^{-1}_{ik}\sum^N_{j=1}A_{ij}X_j\\
        &=\sum^N_{j=1}D^{-1}_{ii}A_{ij}X_j\\
        &=\sum^N_{j=1}\frac{A_{ij}}{D_{ii}}X_j\\
        &=\sum^N_{j=1}\frac{A_{ij}}{\sum^N_{j=1}A_{ik}}X_j
\end{aligned}
$$

其实就是以边为权值对节点特征求加权平均。其中,可以理解为卷积核。如果不了解图卷积可以看[这里](https://www.zhihu.com/question/54504471/answer/611222866)。

![归一化的加权平均法](image\如何评价ST-GCN动作识别算法\归一化的加权平均法.html)

Multi-Kernal考虑到动作识别的特点，作者并未使用单一的卷积核，而是使用『图划分』，将  $\hat{A}$分解成了 $\hat{A_1},\hat{A_2},\hat{A_3}$ 。（作者其实提出了几种不同的图划分策略，但是只有这个比较好用）

![原始图](如何评价ST-GCN动作识别算法\image\原始图.html)


