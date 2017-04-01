# 局部连接层LocallyConnceted

## LocallyConnected1D层
```python
keras.layers.local.LocallyConnected1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```
```LocallyConnected1D```层与```Convolution1D```工作方式类似，唯一的区别是不进行权值共享。即施加在不同输入patch的滤波器是不一样的，当使用该层作为模型首层时，需要提供参数```input_dim```或```input_shape```参数。参数含义参考```Convolution1D```。注意该层的```input_shape```必须完全指定，不支持```None```

### 参数

* nb_filter：卷积核的数目（即输出的维度）

* filter_length：卷积核的空域或时域长度

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”或“same”

* subsample_length：输出对输入的下采样因子

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[<font color='FF0000'>ActivityRegularizer</font>](../other/regularizers)对象

* W_constraints：施加在权重上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* b_constraints：施加在偏置上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* bias：布尔值，是否包含偏置向量（即层对输入做线性变换还是仿射变换）

* input_dim：整数，输入数据的维度。当该层作为网络的第一层时，必须指定该参数或```input_shape```参数。

* input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接```Flatten```层，然后又要连接```Dense```层时，需要指定该参数，否则全连接的输出无法计算出来。

### 输入shape

形如（samples，steps，input_dim）的3D张量

### 输出shape

形如（samples，new_steps，nb_filter）的3D张量，因为有向量填充的原因，```steps```的值会改变

### 例子
```python
# apply a unshared weight convolution 1d of length 3 to a sequence with
# 10 timesteps, with 64 output filters
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# now model.output_shape == (None, 8, 64)
# add a new conv1d on top
model.add(LocallyConnected1D(32, 3))
# now model.output_shape == (None, 6, 32)
```

***

## LocallyConnected2D层
```python
keras.layers.local.LocallyConnected2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='default', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```
```LocallyConnected2D```层与```Convolution2D```工作方式类似，唯一的区别是不进行权值共享。即施加在不同输入patch的滤波器是不一样的，当使用该层作为模型首层时，需要提供参数```input_dim```或```input_shape```参数。参数含义参考```Convolution2D```。注意该层的```input_shape```必须完全指定，不支持```None```

### 参数

* nb_filter：卷积核的数目

* nb_row：卷积核的行数

* nb_col：卷积核的列数

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”或“same”

* subsample：长为2的tuple，输出对输入的下采样因子，更普遍的称呼是“strides”

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[<font color='FF0000'>ActivityRegularizer</font>](../other/regularizers)对象

* W_constraints：施加在权重上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* b_constraints：施加在偏置上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

* bias：布尔值，是否包含偏置向量（即层对输入做线性变换还是仿射变换）

### 输入shape

‘th’模式下，输入形如（samples,channels，rows，cols）的4D张量

‘tf’模式下，输入形如（samples，rows，cols，channels）的4D张量

注意这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的```input_shape```，请参考下面提供的例子。

### 输出shape

‘th’模式下，为形如（samples，nb_filter, new_rows, new_cols）的4D张量

‘tf’模式下，为形如（samples，new_rows, new_cols，nb_filter）的4D张量

输出的行列数可能会因为填充方法而改变

### 例子

```python
# apply a 3x3 unshared weights convolution with 64 output filters on a 32x32 image:
model = Sequential()
model.add(LocallyConnected2D(64, 3, 3, input_shape=(3, 32, 32)))
# now model.output_shape == (None, 64, 30, 30)
# notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64 parameters

# add a 3x3 unshared weights convolution on top, with 32 output filters:
model.add(LocallyConnected2D(32, 3, 3))
# now model.output_shape == (None, 32, 28, 28)
```
