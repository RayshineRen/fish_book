[合集 - 深度学习入门-基于Python的理论与实现(鱼书)学习笔记(4)](https://www.cnblogs.com/SteinsGateSg/collections/31852)

[1.深度学习入门-基于Python的理论与实现(鱼书)学习笔记-Chapter 3-神经网络](https://www.cnblogs.com/SteinsGateSg/articles/19093954)

2.深度学习入门-基于Python的理论与实现(鱼书)学习笔记-Chapter 4-神经网络的学习

[3.深度学习入门-基于Python的理论与实现(鱼书)学习笔记-Chapter 5-误差反向传播](https://www.cnblogs.com/SteinsGateSg/articles/19125403) [4.深度学习入门-基于Python的理论与实现(鱼书)学习笔记-Chapter 7-卷积神经网络](https://www.cnblogs.com/SteinsGateSg/articles/19126385)

## Chapter4. 神经网络的学习

`学习` 是指从训练数据中自动获取最优权重参数的过程, `学习的目的` 是尽可能减小损失函数的值

## 4.1 从数据中学习

比如，对于数字图像识别来说，一种方案是，先从图像中提取 `特征量` ，再利用机器学习技术学习这些 `特征量` 的模式。 `特征量` 是指可以从输入数据(输入图像)中准确地提取本质数据(重要的数据)的转换器。特征量通常表示为向量的形式，使用这些特征量将图像数据转换为向量，然后用机器学习方法学习。

在机器学习方法中，由机器从收集到的数据中找到规律，但是特征量仍由人设计，而在神经网络中，特征量也是由机器来学习的神经网络有时又称为端到端机器学习，端到端这里指的是从一端到另外一端的意思，也就是从原始数据(输入)中获得目标结果(输出)。

**训练数据和测试数据**

机器学习的 `最终目标` 是获得 `泛化能力` ， `泛化` 能力指模型处理未被观察过的数据(不在训练集中的数据)的能力，所以要分为训练数据和测试数据

仅用一个数据集去学习和评价参数是无法正确进行评价的，只对某个数据集过度拟合的状态成为 `过拟合`

简单地说：训练数据 $\backslash\text{rArr}$ 课后练习题， 测试数据 $\backslash\text{rArr}$ 期末考试，过拟合 $\backslash\text{rArr}$ 死背答案

## 4.2 损失函数

损失函数是表示神经网络性能地 `恶劣程度` 的指标，即当前的神经网络对监督数据在多大程度上不拟合，在多大程度上不一致

一般使用 `均方误差` 和 `交叉熵误差` 等

### 4.2.1 均方误差

$$
E = \frac{1}{2} \underset{k}{\sum} \left(\right. y_{k} - t_{k} \left.\right)^{2}
$$

$y_{k}$ 表示神经网络的实际输出， $t_{k}$ 表示监督数据， $k$ 表示数据的维数

```python
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)
```

### 4.2.2 交叉熵误差

$$
E = - \underset{k}{\sum} t_{k} l o g y_{k}
$$

$y_{k}$ 是神经网络的输出， $t_{k}$ 是正确解标签， $t_{k}$ 中只有正确解标签的索引的值是1，其他均为0，所以实际上只计算正确解标签的输出的自然对数

```python
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) #避免出现log(0)
```

### 4.2.3 mini-batch学习

机器学习使用训练数据进行学习，计算损失函数时必须将所有的训练数据作为对象，以交叉熵损失函数为例

$$
E = - \frac{1}{N} \underset{i}{\sum} \underset{j}{\sum} t_{i j} l o g y_{i j}
$$

$N$ 表示有 $N$ 个数据， $y_{i j}$ 表示第 $i$ 个数据第 $j$ 个标签的实际输出值， $t_{i j}$ 是第 $i$ 个数据第 $j$ 个标签的正确解标签，正确为1，不正确为0

数据量可能会特别大导致学习时间特别长，因此，我们可以从全部数据中选取一部分，作为全部数据的 `近似` 。神经网络的学习也是从训练数据中选出一批数据(称为 `mini-batch` ， 小批量)，然后对每个 `mini-batch` 进行学习，称为 `mini-batch` 学习

### 4.2.4 mini-batch版交叉熵误差的实现

这里实现一个能同时处理单个数据和批量数据(**数据作为batch集中输入**)两种情况的函数

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
```

这里，y是神经网络的输出，t是监督数据。y的维度为1时，需要改变数据形状，从(x,) $\backslash\text{rArr}$ (1, x)。并且，当输入为mini-batch时，要用batch个数进行正规化，计算单个数据的平均交叉熵误差

此外，当监督数据是标签形式(非one-hot表示，而是像"2"，"7"这样的标签)时，交叉熵误差如下

```python
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
```

最后一段乍一看比较复杂，我们拆开看  
`y[np,arrange(batch_size), t]` 这个是NumPy的列表索引，从左列表取出第一个维度的索引，右边取出第二个维度的索引，也就是说我们只取出了输出正确解标签的数据，然后取对数再求和，和之前 `one-hot` 的交叉熵是一样的

**在 one-hot 编码下，交叉熵损失简化为：取模型预测的“正确类别”的概率，然后计算它的负对数。**

### 4.2.5 为何要设定损失函数

*为什么要引入损失函数呢？ 直接用精确度不行吗？*

举一个例子，在这样一个场景:一个神经网络在100笔训练数据中精确识别出了32笔，此时精度为32%，即使识别精度有所改善，它也只会像33%，34%......这样不连续地变化。而如果把损失函数作为指标，就可以连续变化了。

所以， **在进行神经网络的学习时，不能将识别精度作为指标。因为如果以识别精度为指标，则参数的导数在绝大多数地方都会变成0**

## 4.3 数值微分

### 4.3.1 导数

高等数学中的导数

$$
\frac{d f \left(\right. x \left.\right)}{d x} = \underset{h \rightarrow 0}{lim} \frac{f \left(\right. x + h \left.\right) - f \left(\right. x \left.\right)}{x}
$$

一个实现方法

```python
def numerical_diff(f, x):
    h = 1e-50
    return (f(x + h) - f(x)) / h
```

可以改进的地方：

1. 过小的值无法表示，h要改得大一些
2. h改大了又不符合导数得定义了，因为h不够小，那么就把前向差分改为中心差分

改进的实现

```python
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / 2 * h
```

### 4.3.2 偏导

高等数学中的偏导

$$
f \left(\right. x , x_{0} \left.\right) = x^{2} + x_{0}^{2}
$$
 
$$
\frac{\partial f}{\partial x} = 2 x \frac{\partial f}{\partial x_{0}} = 2 x_{0}
$$

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_250906130405_Figure_1.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_250906130405_Figure_1.png)

## 4.4 梯度

像 $\left(\right. \frac{\partial f}{\partial x} , \frac{\partial f}{\partial x_{0}} \left.\right)$ 这样的由全部变量的偏导数汇总而成的向量称为 `梯度` 。

```python
import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) #和x形状相同的全0数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x + h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 还原
    
    return grad
```

举个例子, $f \left(\right. x , x_{0} \left.\right) = x^{2} + x_{0}^{2}$ 的梯度  
[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_250906130452_Figure_2.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_250906130452_Figure_2.png)  
这里画的是元素值为负梯度的向量(`负梯度方向是梯度法中变量的更新方向`)

**梯度会指向各店处函数值降低的方向**

### 4.4.1 梯度法

通过巧妙地使用梯度来寻找函数的最小值(或者尽可能小的值)的方法就是梯度法

需要注意的是：梯度表示的是各点处的函数值减少最多的方向。因此,无法保证梯度所指的方向就是函数的最小值或者真正应该前进的方向。

函数的极小值，最小值以及 `鞍点(saddle point)` 梯度为0。极小值是局部最小值。鞍点是从某个方向上看是极大值，从另一个方向上看则是极小值的点。虽然梯度法是要寻找梯度为0的地方，但是那个地方不一定就是最小值(也有可能是极小值或者鞍点)。此外，当函数很复杂且呈扁平状时，学习可能进入一个几乎平坦的地区，陷入被称为 `学习高原` 的无法前进的停滞期。

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_250906134800_saddle.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_250906134800_saddle.png)

**虽然梯度的方向并不一定指向最小值，但是沿着它的方向能够最大限度地减小函数的值**

*梯度法*  
在梯度法中，函数的取值从当前位置沿着梯度方向前进一定距离，然后在新的地方重新求梯度，再沿着新梯度的方向前进，如此反复，不断地沿梯度方向前进。像这样，通过不断地沿梯度方向前进，逐渐减小函数值的过程就是梯度法(严格来说叫梯度下降法)

$$
x_{0} = x_{0} - \eta \frac{\partial f}{\partial x_{0}} x_{1} = x_{1} - \eta \frac{\partial f}{\partial x_{1}}
$$

$\eta$ 表示更新量，在神经网络的学习中称为 `学习率`

```python
"""
init_x 初始值
lr 学习率
step_num 梯度法的重复次数
numerical_gradient(f, x) 之前求函数梯度的函数
"""
def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x
```

需要注意的是，学习率过大或者过小都无法得到好的结果。过大会发散成一个很大的值，过小基本没怎么更新就结束了。

像学习率这样的参数称为 `超参数` ，不同于神经网络的参数（权重和偏置）是通过训练数据和学习算法自动获得的，超参数需要人工设定，通常需要尝试多个值

### 4.4.2 神经网络的梯度

神经网络的学习也要求梯度，这里所说的梯度是指损失函数关于权重参数的梯度

$$
W = \left(\right. w_{11} w_{12} w_{13} \\ w_{21} w_{22} w_{23} \left.\right)
$$
 
$$
\frac{\partial L}{\partial W} = \left(\right. \frac{\partial L}{\partial w_{11}} \frac{\partial L}{\partial w_{12}} \frac{\partial L}{\partial w_{13}} \\ \frac{\partial L}{\partial w_{21}} \frac{\partial L}{\partial w_{22}} \frac{\partial L}{\partial w_{23}} \left.\right)
$$

$\frac{\partial L}{\partial W}$ 的元素由各个元素关于W的偏导数构成。 $\frac{\partial L}{\partial W_{11}}$ 表示当 $w_{11}$ 发生微小变化时，损失函数L的变化率。  
$\frac{\partial L}{\partial W}$ 的形状与W相同

一个简单的实现

```python
import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss
```

## 4.5 学习算法的实现

**小总结**  
神经网络存在合适的权重和偏置，调整权重和偏置以便拟合训练数据的过程称为 `学习` 。神经网络的学习分成下面4个步骤

**步骤1(mini-batch)**  
从训练数据中随机选出一部分数据，这部分数据称为 `mini-batch` ，我们的目标是减小mini-batch的损失函数的值

**步骤2(计算梯度)**  
为了减小mini-batch的损失函数的值，需要求出各个权重参数的梯度。梯度表示损失函数的值减小最多的方向

**步骤3(更新参数)**  
将权重参数沿梯度方向进行微小更新

**步骤4(重复)**  
重复步骤1，2，3

### 4.5.1 2层神经的类

```python
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weigh_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weigh_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weigh_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x:输入数据，t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shaoe[0])
        return accuracy
    
    # x:输入数据，t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
```

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_251003083147_4-3.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_251003083147_4-3.png)  
[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_251003083155_4-4.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_251003083155_4-4.png)

### 4.5.2 mini-batch的实现

以为这里计算梯度用的是数值梯度，所以运行极其慢

```python
import os, sys
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from MNIST.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# 超参数
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

print(train_loss_list)
```

因为训练及其慢，这里直接搬了书里面的图

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_251003091925_4-5.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_251003091925_4-5.png)

### 4.5.3 基于测试数据的评价

根据前面的结果，我们可以确认通过反复学习可以使损失函数的值逐渐减小，不过这个损失函数的值，严格地讲是 **对训练数据地某个mini-batch的损失函数** 的值，但是光看这个结果还不能说明该神经网络再其他数据集上也一定有同等程度的表现

这里，我们没经过一个 **epoch** ，都会记录下训练数据和测试数据的识别精度

*epoch是一个单位。一个epoch表示学习中所有训练数据均被使用过一次时的更新次数。比如，对于10000笔的更新数据，用大小为100笔数据的mini-batch进行学习时，重复SGD100次，所有的训练数据就都被"看过"了，此时，100次就是一个epoch*

简单修改前面代码即可

```python
import os, sys
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from MNIST.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []
train_acc_list = []
test_acc_list = []
# 平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

# 超参数
iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    #计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
```

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_251003091931_4-6.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2473489/o_251003091931_4-6.png)

\_\_EOF\_\_

[![](https://pic.cnblogs.com/avatar/3497511/20240916092737.png)](https://pic.cnblogs.com/avatar/3497511/20240916092737.png)

- **本文作者：** [栗悟饭与龟功気波](https://www.cnblogs.com/SteinsGateSg)
- **本文链接：** [https://www.cnblogs.com/SteinsGateSg/articles/19124787](https://www.cnblogs.com/SteinsGateSg/articles/19124787)
- **关于博主：** 评论和私信会在第一时间回复。或者 [直接私信](https://msg.cnblogs.com/msg/send/SteinsGateSg) 我。
- **版权声明：** 除特殊说明外，转载请注明出处～\[知识共享署名-相同方式共享 4.0 国际许可协议\]
- **声援博主：** 如果您觉得文章对您有帮助，可以点击文章右下角 **【推荐】** 一下。