## SpatialDropout1D层

```python
keras.layers.core.SpatialDropout1D(p)
```
SpatialDropout1D与Dropout的作用类似，但它断开的是整个1D特征图，而不是单个神经元。如果一张特征图的相邻像素之间有很强的相关性（通常发生在低层的卷积层中），那么普通的dropout无法正则化其输出，否则就会导致明显的学习率下降。这种情况下，SpatialDropout1D能够帮助提高特征图之间的独立性，应该用其取代普通的Dropout

### 参数

* p：0~1的浮点数，控制需要断开的链接的比例

### 输入shape

输入形如（samples，timesteps，channels）的3D张量

### 输出shape

与输入相同

### 参考文献

* [<font color='FF0000'>Efficient Object Localization Using Convolutional Networks</font>](https://arxiv.org/pdf/1411.4280.pdf)


***
## SpatialDropout2D层

```python
keras.layers.core.SpatialDropout2D(p, dim_ordering='default')
```
SpatialDropout2D与Dropout的作用类似，但它断开的是整个2D特征图，而不是单个神经元。如果一张特征图的相邻像素之间有很强的相关性（通常发生在低层的卷积层中），那么普通的dropout无法正则化其输出，否则就会导致明显的学习率下降。这种情况下，SpatialDropout2D能够帮助提高特征图之间的独立性，应该用其取代普通的Dropout

### 参数

* p：0~1的浮点数，控制需要断开的链接的比例
* dim_ordering:'th'或'tf'，默认为```~/.keras/keras.json```配置的```image_dim_ordering```值

### 输入shape

‘th’模式下，输入形如（samples，channels，rows，cols）的4D张量

‘tf’模式下，输入形如（samples，rows，cols，channels）的4D张量

注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的```input_shape```，请参考下面提供的例子。

### 输出shape

与输入相同

### 参考文献

* [<font color='FF0000'>Efficient Object Localization Using Convolutional Networks</font>](https://arxiv.org/pdf/1411.4280.pdf)

***

## SpatialDropout3D层

```python
keras.layers.core.SpatialDropout3D(p, dim_ordering='default')
```
SpatialDropout3D与Dropout的作用类似，但它断开的是整个3D特征图，而不是单个神经元。如果一张特征图的相邻像素之间有很强的相关性（通常发生在低层的卷积层中），那么普通的dropout无法正则化其输出，否则就会导致明显的学习率下降。这种情况下，SpatialDropout3D能够帮助提高特征图之间的独立性，应该用其取代普通的Dropout

### 参数

* p：0~1的浮点数，控制需要断开的链接的比例
* dim_ordering:'th'或'tf'，默认为```~/.keras/keras.json```配置的```image_dim_ordering```值

### 输入shape

‘th’模式下，输入应为形如（samples，channels，input_dim1，input_dim2, input_dim3）的5D张量

‘tf’模式下，输入应为形如（samples，input_dim1，input_dim2, input_dim3，channels）的5D张量

### 输出shape

与输入相同

### 参考文献

* [<font color='FF0000'>Efficient Object Localization Using Convolutional Networks</font>](https://arxiv.org/pdf/1411.4280.pdf)

***

***

## TimeDisributedDense层
```python
keras.layers.core.TimeDistributedDense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```
为输入序列的每个时间步信号（即维度1）建立一个全连接层，当RNN网络设置为```return_sequence=True```时尤其有用

* 注意：该层已经被弃用，请使用其包装器```TImeDistributed```完成此功能

```python
model.add(TimeDistributed(Dense(32)))
```

### 参数

* output_dim：大于0的整数，代表该层的输出维度。模型中非首层的全连接层其输入维度可以自动推断，因此非首层的全连接定义时不需要指定输入维度。

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[<font color='FF0000'>ActivityRegularizer</font>](../other/regularizers)对象

* W_constraints：施加在权重上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* b_constraints：施加在偏置上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* bias：布尔值，是否包含偏置向量（即层对输入做线性变换还是仿射变换）

* input_dim：整数，输入数据的维度。当该层作为网络的第一层时，必须指定该参数或```input_shape```参数。

* input_length：输入序列的长度，为整数或None，若为None则代表输入序列是变长序列

### 输入shape

形如 ```(nb_sample, time_dimension, input_dim)```的3D张量

### 输出shape

形如 ```(nb_sample, time_dimension, output_dim)```的3D张量
