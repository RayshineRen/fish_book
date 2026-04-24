[合集 - 深度学习入门-基于Python的理论与实现(鱼书)学习笔记(4)](https://www.cnblogs.com/SteinsGateSg/collections/31852)

1.深度学习入门-基于Python的理论与实现(鱼书)学习笔记-Chapter 3-神经网络

[2.深度学习入门-基于Python的理论与实现(鱼书)学习笔记-Chapter 4-神经网络的学习](https://www.cnblogs.com/SteinsGateSg/articles/19124787) [3.深度学习入门-基于Python的理论与实现(鱼书)学习笔记-Chapter 5-误差反向传播](https://www.cnblogs.com/SteinsGateSg/articles/19125403) [4.深度学习入门-基于Python的理论与实现(鱼书)学习笔记-Chapter 7-卷积神经网络](https://www.cnblogs.com/SteinsGateSg/articles/19126385)

## Chapter 3. 神经网络

## 3.1 从感知机到神经网络

神经网络分为三层： `输入层`, `中间层(也称隐藏层)`, `输出层`)

实例中的网络一共有三层，但是只有两层有权重，所以也成为 `两层网络`, 但是有的书也看层数将其成为 `三层网络`

接下来观察一个感知机

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915142907_3-2.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915142907_3-2.png)

当然这里 `偏置b` 没有表现出来，也可以把 `偏置b` 当成一个 **信号恒为1, 权重为b的神经元**

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915142928_3-3.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915142928_3-3.png)

用数学式表示

$$
y = \left{\right. 0 & \left(\right. b + w_{1} x_{1} + w_{2} x_{2} \leq 0 \left.\right) \\ 1 & \left(\right. b + x_{1} x_{1} + w_{2} x_{2} > 0 \left.\right)
$$

可以写得更简洁一些

$$
y = h \left(\right. b + w_{1} x_{1} + w_{2} x_{2} \left.\right)
$$
 
$$
h \left(\right. x \left.\right) = \left{\right. 0 & x \leq 0 \\ 1 & x > 0
$$

**引出激活函数**  
刚刚的 $h \left(\right. x \left.\right)$ 将输入信号的总和转换为输出信号，这样的函数称为激活函数，作用是决定如何来激活输入信号的总和

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143145_3-4.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143145_3-4.png)

## 3.2 激活函数

### 3.2.1 sigmoid函数

$$
h \left(\right. x \left.\right) = \frac{1}{1 + e^{- x}}
$$

```python
def sigmoid(x):
    return 1.0 / (1 + numpy.exp(-x))
```

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143213_Figure_1.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143213_Figure_1.png))

### 3.2.2 阶跃函数

```python
def step_function(x):
    if (x > 0):
        return 1
    else:
        return 0
```

为了方便后面的操作，可以把它修改成为支持Numpy数组的实现

```python
def step_function(x):
    y = x > 0
    return y.astype(np.int64)
```

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143234_Figure_2.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143234_Figure_2.png)

**注意** 阶跃函数和sigmoid函数都是非线性函数，神经网络的激活函数必须使用非线性函数，使用线性函数的缺点在于:`不管如何加深层数，总是存在与之等效的"无隐藏层的神经网络"` ， 这样的话加深神经网络层数就没有意义了

### 3.2.3 ReLU函数

$$
h \left(\right. x \left.\right) = \left{\right. x & \left(\right. x > 0 \left.\right) \\ 0 & \left(\right. x \leq 0 \left.\right)
$$

```python
def relu(x):
    return np.maximum(0, x)
```

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143316_Figure_3.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143316_Figure_3.png)

## 3.3 多维数组的运算

### 3.3.1 多维数组

np.dim() 返回数组的维数  
np.shape返回数组的形状，类型为 `tuple`

### 3.3.2 矩阵乘法

这里主要是线性代数的知识

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143338_3-11.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143338_3-11.png)

Numpy中的矩阵乘法

np.dot(matA, matB)

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)
```

**注意**  
矩阵乘法和广播计算不同

### 3.3.3 神经网络的内积

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143359_3-14.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143359_3-14.png)

```python
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
```

## 3.4 三层神经网络的实现

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143423_3-15.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143423_3-15.png)

输入层(第0层)有2个神经元  
第1个隐层层(第1层)有3个神经元  
第2个隐藏层(第2层)有2个神经元  
输出层(第3层)有2个神经元

**一些符号的规定**

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143447_3-16.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143447_3-16.png)

**传递过程**

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143509_3-17.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143509_3-17.png)

$$
a_{1}^{\left(\right. 1 \left.\right)} = w_{11}^{\left(\right. 1 \left.\right)} x_{1} + w_{12}^{\left(\right. 1 \left.\right)} x_{2} + b_{1}^{\left(\right. 1 \left.\right)} a_{2}^{\left(\right. 1 \left.\right)} = w_{21}^{\left(\right. 1 \left.\right)} x_{2} + w_{22}^{\left(\right. 1 \left.\right)} x_{2} + b_{2}^{\left(\right. 1 \left.\right)} a_{3}^{\left(\right. 1 \left.\right)} = w_{31}^{\left(\right. 1 \left.\right)} x_{3} + w_{32}^{\left(\right. 1 \left.\right)} x_{2} + b_{3}^{\left(\right. 1 \left.\right)}
$$

用矩阵表示就是

$$
A^{\left(\right. 1 \left.\right)} = \left(\right. a_{1}^{\left(\right. 1 \left.\right)} , a_{2}^{\left(\right. 1 \left.\right)} , a_{3}^{\left(\right. 1 \left.\right)} \left.\right)
$$
 
$$
X = \left(\right. x_{1} , x_{2} \left.\right)
$$
 
$$
W^{\left(\right. 1 \left.\right)} = \left(\right. w_{11}^{\left(\right. 1 \left.\right)} , w_{21}^{\left(\right. 1 \left.\right)} , w_{31}^{\left(\right. 1 \left.\right)} \\ w_{12}^{\left(\right. 1 \left.\right)} , w_{22}^{\left(\right. 1 \left.\right)} , w_{32}^{\left(\right. 1 \left.\right)} \left.\right)
$$
 
$$
B^{\left(\right. 1 \left.\right)} = \left(\right. b_{1}^{\left(\right. 1 \left.\right)} , b_{2}^{\left(\right. 1 \left.\right)} , b_{3}^{\left(\right. 1 \left.\right)} \left.\right)
$$

然后就是这个式子

$$
A^{\left(\right. 1 \left.\right)} = X W^{\left(\right. 1 \left.\right)} + B^{\left(\right. 1 \left.\right)}
$$

```python
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
```

之后再加上激活函数 `sigmoid` 可以实现 `从输入层到第1层的传递`

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143535_3-18.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143535_3-18.png)

```python
Z1 = sigmoid(A1)
```

接着实现 `从第1层到第2层的传递`

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143556_3-19.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143556_3-19.png)

```python
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
```

最后是从 `第2层到输出层的传递` ， 这里激活函数有些不一样

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143631_3-20.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143631_3-20.png)

```python
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
```

**注意**  
一般地，回归问题可以使用恒等函数，二分类问题可以使用sigmoid函数，多元分类问题可以使用softmax函数

**代码实现小结**

## 3.5 输出层的设计

神经网络可以用在分类问题和回归问题上，不过需要根据情况改变输出层的激活函数

**恒等函数**  
输入信号原封不动地被输出

```python
def identify_function(x):
    return x
```

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143658_3-21.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143658_3-21.png)

**softmax函数**

$$
y_{k} = \frac{e^{a_{k}}}{\sum_{i = 1}^{n} e^{\left(\right. a_{i} \left.\right)}}
$$

表示：假设输出层一共有n个神经元，计算第k个神经元的输出 $y_{k}$ ，softmax函数的分子是输入信号 $a_{k}$ 的指数函数，分母是所有输入信号的指数函数的和

```python
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
```

[![](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143720_3-22.png)](https://images.cnblogs.com/cnblogs_com/blogs/828475/galleries/2474880/o_250915143720_3-22.png)

**softmax函数需要注意的事**

由于指数爆炸，所有数可能会非常打，所以要做一些变换

$$
y_{k} = \frac{e^{a_{k}}}{\sum_{i = 1}^{n} e^{\left(\right. a_{i} \left.\right)}} = \frac{C e^{a_{k}}}{C \sum_{i = 1}^{n} e^{\left(\right. a_{i} \left.\right)}} = \frac{e^{a_{k} + l o g C}}{\sum_{i = 1}^{n} e^{\left(\right. a_{i} + l o g C \left.\right)}} = \frac{e^{a_{k} + C^{`}}}{\sum_{i = 1}^{n} e^{\left(\right. a_{i} + C^{`} \left.\right)}}
$$

```python
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
```

**softmax函数的特征**

- 1- softmax函数总是输出0.0到1.0之间的实数
- 2 -softmax函数输出值的总和是1，所以才可以把softmax函数的输出解释为"概率"
- 3 -即使使用了softmax函数，各个元素之间的大小关系也不会改变

*求解机器学习问题的步骤可以分为“学习”和“推理”两个阶段，首先，在学习阶段进行模型的学习，然后，在推理阶段，用学到的模型对未知的数据进行推理（分类）。推理阶段一般都会省略输出层的softmax函数。在输出层使用softmax函数是因为它和神经网络的学习有关*

**输出层的神经元的数量**

输出层的神经元的数量需要根据待解决的问题来决定。对于分类问题，输出层的神经元数量一般设定为类别的数量

\_\_EOF\_\_

[![](https://pic.cnblogs.com/avatar/3497511/20240916092737.png)](https://pic.cnblogs.com/avatar/3497511/20240916092737.png)

- **本文作者：** [栗悟饭与龟功気波](https://www.cnblogs.com/SteinsGateSg)
- **本文链接：** [https://www.cnblogs.com/SteinsGateSg/articles/19093954](https://www.cnblogs.com/SteinsGateSg/articles/19093954)
- **关于博主：** 评论和私信会在第一时间回复。或者 [直接私信](https://msg.cnblogs.com/msg/send/SteinsGateSg) 我。
- **版权声明：** 除特殊说明外，转载请注明出处～\[知识共享署名-相同方式共享 4.0 国际许可协议\]
- **声援博主：** 如果您觉得文章对您有帮助，可以点击文章右下角 **【推荐】** 一下。