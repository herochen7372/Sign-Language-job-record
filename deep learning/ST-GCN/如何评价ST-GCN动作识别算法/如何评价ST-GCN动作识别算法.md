# 如何评价ST-GCN动作识别算法？

[Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition[paperwithcode]](https://arxiv.org/abs/1801.07455)

作者的主要工作就两点：
- 使用 OpenPose 处理了视频，提出了一个数据集
- 结合 GCN 和 TCN 提出了模型，在数据集上效果还不错

## OpenPose 预处理
&emsp;&emsp;OpenPose 是一个标注人体的关节（颈部，肩膀，肘部等），连接成骨骼，进而估计人体姿态的算法。作为视频的预处理工具，我们只需要关注 OpenPose 的输出就可以了。

![视频中的骨骼标注](image/视频中的骨骼标注.html)

&emsp;&emsp;总的来说，视频的骨骼标注结果维数比较高。在一个视频中，可能有很多帧（Frame）。每个帧中，可能存在很多人（Man）。每个人又有很多关节（Joint）。每一个关节又有不同特征（位置、置信度）。

![关节的特征](image/关节的特征.html)

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

![ST-GCN 网络结构](image/ST-GCN网络结构.html)

主要分为三部分：

![归一化](image/关节的特征.html)

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

![时空变换](image/时空变换.html)

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

![read out 输出](image/readout.html)

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

![归一化的加权平均法](image/归一化的加权平均法.html)

Multi-Kernal考虑到动作识别的特点，作者并未使用单一的卷积核，而是使用『图划分』，将  $\hat{A}$分解成了 $\hat{A_1},\hat{A_2},\hat{A_3}$ 。（作者其实提出了几种不同的图划分策略，但是只有这个比较好用）

![原始图](image/原始图.html)


&emsp;&emsp;$\hat{A}$表示的所有边如上图右侧所示：

- 两个节点之间有一条双向边
- 节点自身有一个自环

&emsp;&emsp;作者结合运动分析研究，将其划分为三个子图，分别表达向心运动、离心运动和静止的动作特征。

![子图划分方法](image/子图划分方法.html)

&emsp;&emsp;对于一个根节点，与它相连的边可以分为 3 部分。
- 第 1 部分连接了空间位置上比本节点更远离整个骨架重心的邻居节点（黄色节点），包含了离心运动的特征。
- 第 2 部分连接了更为靠近重心的邻居节点（蓝色节点），包含了向心运动的特征。
- 第 3 部分连接了根节点本身（绿色节点），包含了静止的特征。

![子图划分结果](image/子图划分结果.html)

&emsp;&emsp;使用这样的分解方法，1 个图分解成了 3 个子图。卷积核也从 1 个变为了 3 个，即 (1,18,18)变为(3,18,18) 。3 个卷积核的卷积结果分别表达了不同尺度的动作特征。要得到卷积的结果，只需要使用每个卷积核分别进行卷积，在进行加权平均（和图像卷积相同）。

&emsp;&emsp;具体实现如下：
```
A = []
for hop in valid_hop:
    a_root = np.zeros((self.num_node, self.num_node))
    a_close = np.zeros((self.num_node, self.num_node))
    a_further = np.zeros((self.num_node, self.num_node))
    for i in range(self.num_node):
        for j in range(self.num_node):
            if self.hop_dis[j, i] == hop:
                if self.hop_dis[j, self.center] == self.hop_dis[
                        i, self.center]:
                    a_root[j, i] = normalize_adjacency[j, i]
                elif self.hop_dis[j, self.
                                  center] > self.hop_dis[i, self.
                                                         center]:
                    a_close[j, i] = normalize_adjacency[j, i]
                else:
                    a_further[j, i] = normalize_adjacency[j, i]
    if hop == 0:
        A.append(a_root)
    else:
        A.append(a_root + a_close)
        A.append(a_further)
A = np.stack(A)
self.A = A
```

&emsp;&emsp;Multi-Kernal GCN现在，我们可以写出带有k个卷积核的图卷积表达式了： 
$$\sum_{k}\sum_{v}{(XW)}_{nkctv}\hat{A}_{kvw}=X'_{nctw}$$
&emsp;&emsp;表达式可以用爱因斯坦求和约定表示 nkctv,kvw→nctw。其中，
- n 表示所有视频中的人数（batch * man）
- k 表示卷积核数（使用上面的分解方法 k=3）
- c 表示关节特征数（64 ... 128）
- t 表示关键帧数（150 ... 38）
- v 和 w 表示关节数（使用 OpenPose 的话有 18 个节点）

&emsp;&emsp;对 v 求和代表了节点的加权平均，对 k 求和代表了不同卷积核 feature map 的加权平均，具体实现如下：

```
# self.conv = nn.Conv2d(
#             in_channels,
#             out_channels * kernel_size,
#             kernel_size=(t_kernel_size, 1),
#             padding=(t_padding, 0),
#             stride=(t_stride, 1),
#             dilation=(t_dilation, 1),
#             bias=bias)

x = self.conv(x)
n, kc, t, v = x.size()
x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
x = torch.einsum('nkctv,kvw->nctw', (x, A))
return x.contiguous(), A
```

&emsp;&emsp;如果要类比的话，其实和 GoogleNet 的思路有些相似：

都在一个卷积单元中试图利用不同感受野的卷积核，提取不同分量的特征。

![GoogleNet](image/GoogleNet.html)

## TCN
&emsp;&emsp;GCN 帮助我们学习了到空间中相邻关节的局部特征。在此基础上，我们需要学习时间中关节变化的局部特征。如何为 Graph 叠加时序特征，是图网络面临的问题之一。这方面的研究主要有两个思路：时间卷积（TCN）和序列模型（LSTM）。 

&emsp;&emsp;ST-GCN 使用的是 TCN，由于形状固定，我们可以使用传统的卷积层完成时间卷积操作。为了便于理解，可以类比图像的卷积操作。st-gcn 的 feature map 最后三个维度的形状为 (C,V,T) ，与图像 feature map 的形状 (C,W,H)相对应。
- 图像的通道数 C 对应关节的特征数 C 。
- 图像的宽 W 对应关键帧数 V 。
- 图像的高 H 对应关节数 T 。

&emsp;&emsp;在图像卷积中，卷积核的大小为$『w』\times『1』$，则每次完成 w 行像素，1 列像素的卷积。『stride』为 s，则每次移动 s 像素，完成 1 行后进行下 1 行像素的卷积。

![时间卷积示意图](image/时间卷积示意图.html)

&emsp;&emsp;在时间卷积中，卷积核的大小为『temporal_kernel_size』 $\times $『1』，则每次完成 1 个节点，temporal_kernel_size 个关键帧的卷积。『stride』为 1，则每次移动 1 帧，完成 1 个节点后进行下 1 个节点的卷积。

&emsp;&emsp;具体实现如下：
```
padding = ((kernel_size[0] - 1) // 2, 0)

self.tcn = nn.Sequential(
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(
        out_channels,
        out_channels,
        (temporal_kernel_size, 1),
        (1, 1),
        padding,
    ),
    nn.BatchNorm2d(out_channels),
    nn.Dropout(dropout, inplace=True),
)
```

&emsp;&emsp;再列举几个序列模型的相关工作，感兴趣的同学可以尝试一下：
- [AGC-Seq2Seq](https://arxiv.org/ftp/arxiv/papers/1810/1810.10237.pdf) 使用的是 Seq2Seq + Attention。
- ST-MGCN 使用的是 CGRNN。
- [DCRNN](https://arxiv.org/pdf/1707.01926.pdf) 使用的是 GRU。

## Attention
&emsp;&emsp;作者在进行图卷积之前，还设计了一个简易的注意力模型（ATT）。如果不了解图注意力模型可以看[这里](https://www.zhihu.com/question/275866887/answer/627791230)。

```
# 注意力参数
# 每个 st-gcn 单元都有自己的权重参数用于训练
self.edge_importance = nn.ParameterList([
    nn.Parameter(torch.ones(self.A.size()))
    for i in self.st_gcn_networks
])
# st-gcn 卷积
for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
    print(x.shape)
    # 关注重要的边信息
    x, _ = gcn(x, self.A * importance)
```

&emsp;&emsp;其实很好理解，在运动过程中，不同的躯干重要性是不同的。例如腿的动作可能比脖子重要，通过腿部我们甚至能判断出跑步、走路和跳跃，但是脖子的动作中可能并不包含多少有效信息。

&emsp;&emsp;因此，ST-GCN 对不同躯干进行了加权（每个 st-gcn 单元都有自己的权重参数用于训练）。



```
作者：日知
链接：https://www.zhihu.com/question/276101856/answer/638672980
来源：知乎
```




