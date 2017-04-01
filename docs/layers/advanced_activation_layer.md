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

### 参考文献

[Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

***

## PReLU层
```python
keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```
该层为参数化的ReLU（Parametric ReLU），表达式是：```f(x) = alpha * x for x < 0```, ```f(x) = x for x>=0```，此处的```alpha```为一个与xshape相同的可学习的参数向量。

### 参数

* alpha_initializer：alpha的初始化函数
* alpha_regularizer：alpha的正则项
* alpha_constraint：alpha的约束项
* shared_axes：该参数指定的轴将共享同一组科学系参数，例如假如输入特征图是从2D卷积过来的，具有形如`(batch, height, width, channels)`这样的shape，则或许你会希望在空域共享参数，这样每个filter就只有一组参数，设定`shared_axes=[1,2]`可完成该目标

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

### 参考文献

* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/pdf/1502.01852v1.pdf)

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

* [>Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/pdf/1511.07289v1.pdf)

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

* [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/pdf/1402.3337.pdf)

***