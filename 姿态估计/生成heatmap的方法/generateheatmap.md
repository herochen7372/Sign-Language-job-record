# 人体姿态估计-生成heatmap的方法

有时间整理下生成heatmap的方法~🙌
主要是两类方便快速的方法，以前的方法这里不讨论

这里假设heatmap大小为(64,48)，关键点的坐标为(32,24)
即将一个(64,48)的黑图的中心点点亮

## 1、利用CV2函数，利用高斯模糊函数生成heatmap
**函数：cv2.GaussianBlur(heatmap, kernel, sigma)**
参数说明：
heatmap：要进行高斯模糊的原图像img
kernel： 高斯核大小 ，一般为正数和奇数
sigma：标准差 ，图像为二维，值为一个数表明x，y方向使用统一的标准差

```
import matplotlib.pyplot as plt
import numpy as np
import cv2

def generate_heatmap(heatmap, sigma):

    heatmap[32][24] = 1
    heatmap = cv2.GaussianBlur(heatmap, sigma, 2)
    am = np.amax(heatmap)
    heatmap /= am / 255
    return heatmap


target = np.zeros((64, 48))
plt.imshow(target, cmap='hot', interpolation='nearest')
plt.show()
target = generate_heatmap(target, (15, 15))
plt.imshow(target, cmap='hot', interpolation='nearest')
plt.show()
print(target)
```

高斯核大小为(15,15)的结果：

![原图](image/1.原图.html)

原图

![高斯模糊后的heatmap](image/1.高斯模糊后的heatmap.html)

高斯模糊后的heatmap


## 2、自定义高斯分布范围生成heatmap，简单快捷有效
与以前对每一个像素进行高斯模糊不同，这里只对3倍的Sigma的区域进行高斯分布(为什么是3sigma，我理解的是高斯分布数值%99.8的概率会落在-3sigma-3sigma区域)，加快了计算速度。

```
import matplotlib.pyplot as plt
import numpy as np
import cv2
def generate_heatmap(heatmap, sigma):

    shape = heatmap.shape
    print(shape)
    tmp_size = sigma * 3           # 这里控制高斯核大小，可改为你想要的高斯核大小
    mu_x = 24            # x坐标
    mu_y = 32            # y坐标
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]           # 关键点高斯分布左上角坐标，[18，26］
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]   # 关键点高斯分布右下角坐标，[31，39］
    size = 2 * tmp_size + 1              # 高斯核大小
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2       # 6
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], shape[0]) - ul[0]   # (0,13) 差值为2 * tmp_size + 1，即size大小
    g_y = max(0, -ul[1]), min(br[1], shape[1]) - ul[1]   # (0,13)
    img_x = max(0, ul[0]), min(br[0], shape[0])   # (26,39)
    img_y = max(0, ul[1]), min(br[1], shape[1])   # (18,31)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    am = np.amax(heatmap)
    heatmap /= am / 255
    return heatmap


target = np.zeros((64, 48))    # h，w
sigma = 2
plt.imshow(target, cmap='hot', interpolation='nearest')
plt.show()
target = generate_heatmap(target, sigma)
plt.imshow(target, cmap='hot', interpolation='nearest')
plt.show()                # 显示图像
print(target)
```
将 tmp_size改为 tmp_size=7，此时高斯核大小为(15,15),由size = 2 * tmp_size + 1 公式计算高斯核大小结果为：

![原图](image/2.原图.html)

原图

![原图](image/2.经过高斯模糊后的heatmap.html)

经过高斯模糊后的heatmap（与方法1的结果相同）
