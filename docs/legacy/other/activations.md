# 激活函数Activations

激活函数可以通过设置单独的[<font color='#FF0000'>激活层</font>](../layers/core_layer/#activation)实现，也可以在构造层对象时通过传递```activation```参数实现。

```python
from keras.layers.core import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

等价于

```python
model.add(Dense(64, activation='tanh'))
```

也可以通过传递一个逐元素运算的Theano/TensorFlow函数来作为激活函数：
```python
from keras import backend as K

def tanh(x):
    return K.tanh(x)

model.add(Dense(64, activation=tanh))
model.add(Activation(tanh)
```

***

## 预定义激活函数

* softmax：对输入数据的最后一维进行softmax，输入数据应形如```(nb_samples, nb_timesteps, nb_dims)```或```(nb_samples,nb_dims)```

* softplus

* softsign

* relu

* tanh

* sigmoid

* hard_sigmoid

* linear

## 高级激活函数

对于简单的Theano/TensorFlow不能表达的复杂激活函数，如含有可学习参数的激活函数，可通过[<font color='#FF0000'>高级激活函数</font>](../layers/advanced_activation_layer)实现，如PReLU，LeakyReLU等

【Tips】待会儿（大概几天吧）我们将把各个激活函数的表达式、图形和特点总结一下。请大家持续关注~

