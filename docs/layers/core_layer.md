# 常用层

常用层对应于core模块，core内部定义了一系列常用的网络层，包括全连接、激活层等

## Dense层
```python
keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
```
Dense就是常用的全连接层，这里是一个使用示例：

```python
# as first layer in a sequential model:
model = Sequential()
model.add(Dense(32, input_dim=16))
# now the model will take as input arrays of shape (*, 16)
# and output arrays of shape (*, 32)

# this is equivalent to the above:
model = Sequential()
model.add(Dense(32, input_shape=(16,)))

# after the first layer, you don't need to specify
# the size of the input anymore:
model.add(Dense(32))
```

### 参数：

* output_dim：大于0的整数，代表该层的输出维度。模型中非首层的全连接层其输入维度可以自动推断，因此非首层的全连接定义时不需要指定输入维度。

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* W_regularizer：施加在权重上的正则项，为[WeightRegularizer](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[WeightRegularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[ActivityRegularizer](../other/regularizers)对象

* W_constraints：施加在权重上的正则项，为[constraints](../other/constraints)对象

* b_constraints：施加在偏置上的正则项，为[constraints](../other/constraints)对象

* bias：布尔值，是否包含偏置向量（即层对输入做线性变换还是仿射变换）

* input_dim：整数，输入数据的维度。当Dense层作为网络的第一层时，必须指定该参数或```input_shape```参数。

### 输入

形如（nb_samples, input_dim）的2D张量

### 输出

形如 （nb_samples, output_dim）的2D张量

***

## Activation层

```python
keras.layers.core.Activation(activation)
```
激活层对一个层的输出施加激活函数

### 参数

* activation：将要使用的激活函数，为预定义激活函数名或一个Tensorflow/Theano的函数。参考[<font color='#FF0000'>激活函数</font>](../other/activations)

### 输入形状

任意，当使用激活层作为第一层时，要指定```input_shape```

### 输出形状

与输入形状相同

***

## Dropout层

```python
keras.layers.core.Dropout(p)
```
为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开p%的输入神经元连接，Dropout层用于防止过拟合。

### 参数

* p：0~1的浮点数，控制需要断开的链接的比例

### 参考文献

* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

***

## Flatten层
```python
keras.layers.core.Flatten()
```
Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。

### 例子
```python
model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 32, 32)))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

***

## Reshape层
```python
keras.layers.core.Reshape(target_shape)
```
Reshape层用来将输入形状转换为特定的形状

## 参数

* target_shape：目标形状，为整数的tuple，不包含样本数目的维度（batch大小）

### 输入形状

任意，但输入的形状必须固定。当使用该层为模型首层时，需要指定```input_shape```参数

### 输出形状

```(batch_size,)+target_shape```

### 例子

```python
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)
```

***

## Permute层
```python
keras.layers.core.Permute(dims)
```
Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。

### 参数

* dims：整数tuple，指定重排的模式，不包含样本数的维度。重拍模式的下标从1开始。例如（2，1）代表将输入的第二个维度重拍到输出的第一个维度，而将输入的第一个维度重排到第二个维度

### 例子

```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```

### 输入形状

任意，当使用激活层作为第一层时，要指定```input_shape```

### 输出形状

与输入相同，但是其维度按照指定的模式重新排列

***

## RepeatVector层
```python
keras.layers.core.RepeatVector(n)
```
RepeatVector层将输入重复n次

### 参数

* n：整数，重复的次数

### 输入形状

形如（nb_samples, features）的2D张量

### 输出形状

形如（nb_samples, n, features）的3D张量

### 例子
```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

***

## Merge层
```python
keras.engine.topology.Merge(layers=None, mode='sum', concat_axis=-1, dot_axes=-1, output_shape=None, node_indices=None, tensor_indices=None, name=None)
```
Merge层根据给定的模式，将一个张量列表中的若干张量合并为一个单独的张量

### 参数

* layers：该参数为Keras张量的列表，或Keras层对象的列表。该列表的元素数目必须大于1。
* mode：合并模式，为预定义合并模式名的字符串或lambda函数或普通函数，如果为lambda函数或普通函数，则该函数必须接受一个张量的list作为输入，并返回一个张量。如果为字符串，则必须是下列值之一：
	* “sum”，“mul”，“concat”，“ave”，“cos”，“dot”

































