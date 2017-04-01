# 高级激活层Advanced Activation

## LeakyReLU层
```python
keras.layers.advanced_activations.LeakyReLU(alpha=0.3)
```
LeakyRelU是修正线性单元（Rectified Linear Unit，ReLU）的特殊版本，当不激活时，LeakyReLU仍然会有非零输出值，从而获得一个小梯度，避免ReLU可能出现的神经元“死亡”现象。即，```f(x)=alpha * x for x < 0```, ```f(x) = x for x>=0```

### 参数

* alpha：大于0的浮点数，代表激活函数图像中第三象限线段的斜率

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

***

## PReLU层
```python
keras.layers.advanced_activations.PReLU(init='zero', weights=None, shared_axes=None)
```
该层为参数化的ReLU（Parametric ReLU），表达式是：```f(x) = alpha * x for x < 0```, ```f(x) = x for x>=0```，此处的```alpha```为一个与xshape相同的可学习的参数向量。

### 参数

* init：alpha的初始化函数

* weights：alpha的初始化值，为具有单个numpy array的list

- shared_axes：该参数指定的轴将共享同一组科学系参数，例如假如输入特征图是从2D卷积过来的，具有形如`(batch, height, width, channels)`这样的shape，则或许你会希望在空域共享参数，这样每个filter就只有一组参数，设定`shared_axes=[1,2]`可完成该目标

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

### 参考文献

* [<font color='FF0000'>Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification</font>](http://arxiv.org/pdf/1502.01852v1.pdf)

***

## ELU层
```python
keras.layers.advanced_activations.ELU(alpha=1.0)
```
ELU层是指数线性单元（Exponential Linera Unit），表达式为：
该层为参数化的ReLU（Parametric ReLU），表达式是：```f(x) = alpha * (exp(x) - 1.) for x < 0```, ```f(x) = x for x>=0```

### 参数

* alpha：控制负因子的参数

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

### 参考文献

* [<font color='FF0000'>Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)</font>](http://arxiv.org/pdf/1511.07289v1.pdf)

***

## ParametricSoftplus层
```python
keras.layers.advanced_activations.ParametricSoftplus(alpha_init=0.2, beta_init=5.0, weights=None, shared_axes=None)
```
该层是参数化的Softplus，表达式是：```f(x) = alpha * log(1 + exp(beta * x))```

### 参数

* alpha_init：浮点数，alpha的初始值

* beta_init：浮点数，beta的初始值

* weights：初始化权重，为含有两个numpy array的list

- shared_axes：该参数指定的轴将共享同一组科学系参数，例如假如输入特征图是从2D卷积过来的，具有形如`(batch, height, width, channels)`这样的shape，则或许你会希望在空域共享参数，这样每个filter就只有一组参数，设定`shared_axes=[1,2]`可完成该目标

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

### 参考文献

* [<font color='FF0000'>Inferring Nonlinear Neuronal Computation Based on Physiologically Plausible Inputs</font>](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003143)

***

## ThresholdedReLU层
```python
keras.layers.advanced_activations.ThresholdedReLU(theta=1.0)
```
该层是带有门限的ReLU，表达式是：```f(x) = x for x > theta```,```f(x) = 0 otherwise```

### 参数

* theata：大或等于0的浮点数，激活门限位置

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

### 参考文献

* [<font color='FF0000'>Zero-Bias Autoencoders and the Benefits of Co-Adapting Features</font>](http://arxiv.org/pdf/1402.3337.pdf)

***

***

## SReLU层
```python
keras.layers.advanced_activations.SReLU(t_left_init='zero', a_left_init='glorot_uniform', t_right_init='glorot_uniform', a_right_init='one', shared_axes=None)
```
该层是S形的ReLU

### 参数

* t_left_init：左侧截断初始化函数

* a_left_init：左侧斜率初始化函数

* t_right_init：右侧截断初始化函数

* a_right_init：右侧斜率初始化函数

- shared_axes：该参数指定的轴将共享同一组科学系参数，例如假如输入特征图是从2D卷积过来的，具有形如`(batch, height, width, channels)`这样的shape，则或许你会希望在空域共享参数，这样每个filter就只有一组参数，设定`shared_axes=[1,2]`可完成该目标

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

### 参考文献

* [<font color='FF0000'>Deep Learning with S-shaped Rectified Linear Activation Units</font>](http://arxiv.org/abs/1512.07030)