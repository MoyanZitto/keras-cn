# 优化器optimizers

优化器是编译Keras模型必要的两个参数之一
```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

可以在调用```model.compile()```之前初始化一个优化器对象，然后传入该函数（如上所示），也可以在调用```model.compile()```时传递一个预定义优化器名。在后者情形下，优化器的参数将使用默认值。
```python
# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')
```
## 所有优化器都可用的参数
参数```clipnorm```和```clipvalue```是所有优化器都可以使用的参数,用于对梯度进行裁剪.示例如下:
```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```
```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

## SGD
```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```
随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量

### 参数

* lr：大或等于0的浮点数，学习率

* momentum：大或等于0的浮点数，动量参数

* decay：大或等于0的浮点数，每次更新后的学习率衰减值

* nesterov：布尔值，确定是否使用Nesterov动量

***

## RMSprop
```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
```
除学习率可调整外，建议保持优化器的其他默认参数不变

该优化器通常是面对递归神经网络时的一个良好选择

### 参数

* lr：大或等于0的浮点数，学习率

* rho：大或等于0的浮点数

* epsilon：大或等于0的小浮点数，防止除0错误

***

## Adagrad
```python
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
```
建议保持优化器的默认参数不变

### Adagrad

* lr：大或等于0的浮点数，学习率

* epsilon：大或等于0的小浮点数，防止除0错误

***

## Adadelta
```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
```
建议保持优化器的默认参数不变

### 参数

* lr：大或等于0的浮点数，学习率

* rho：大或等于0的浮点数

* epsilon：大或等于0的小浮点数，防止除0错误

### 参考文献

***

* [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)

## Adam
```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

该优化器的默认值来源于参考文献

### 参数

* lr：大或等于0的浮点数，学习率

* beta_1/beta_2：浮点数， 0<beta<1，通常很接近1

* epsilon：大或等于0的小浮点数，防止除0错误

### 参考文献

* [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

***

## Adamax
```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
```

Adamax优化器来自于Adam的论文的Section7，该方法是基于无穷范数的Adam方法的变体。

默认参数由论文提供

### 参数

* lr：大或等于0的浮点数，学习率

* beta_1/beta_2：浮点数， 0<beta<1，通常很接近1

* epsilon：大或等于0的小浮点数，防止除0错误

### 参考文献

* [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

***

## Nadam

```python
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
```

Nesterov Adam optimizer: Adam本质上像是带有动量项的RMSprop，Nadam就是带有Nesterov 动量的Adam RMSprop

默认参数来自于论文，推荐不要对默认参数进行更改。

### 参数

* lr：大或等于0的浮点数，学习率

* beta_1/beta_2：浮点数， 0<beta<1，通常很接近1

* epsilon：大或等于0的小浮点数，防止除0错误

### 参考文献

* [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)

* [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

## TFOptimizer
```python
keras.optimizers.TFOptimizer(optimizer)
```
TF优化器的包装器
