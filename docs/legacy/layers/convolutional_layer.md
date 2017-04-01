# 卷积层

## Convolution1D层
```python
keras.layers.convolutional.Convolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

一维卷积层，用以在一维输入信号上进行邻域滤波。当使用该层作为首层时，需要提供关键字参数```input_dim```或```input_shape```。例如```input_dim=128```长为128的向量序列输入，而```input_shape=(10,128)```代表一个长为10的128向量序列

### 参数

* nb_filter：卷积核的数目（即输出的维度）

* filter_length：卷积核的空域或时域长度

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”, “same” 或“full”，full需要以theano为后端

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
# apply a convolution 1d of length 3 to a sequence with 10 timesteps,
# with 64 output filters
model = Sequential()
model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
# now model.output_shape == (None, 10, 64)

# add a new conv1d on top
model.add(Convolution1D(32, 3, border_mode='same'))
# now model.output_shape == (None, 10, 32)
```

【Tips】可以将Convolution1D看作Convolution2D的快捷版，对例子中（10，32）的信号进行1D卷积相当于对其进行卷积核为（filter_length, 32）的2D卷积。【@3rduncle】

***
## AtrousConvolution1D层
```python
keras.layers.convolutional.AtrousConvolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, atrous_rate=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```

AtrousConvolution1D层用于对1D信号进行滤波，是膨胀/带孔洞的卷积。当使用该层作为首层时，需要提供关键字参数```input_dim```或```input_shape```。例如```input_dim=128```长为128的向量序列输入，而```input_shape=(10,128)```代表一个长为10的128向量序列.

### 参数

* nb_filter：卷积核的数目（即输出的维度）

* filter_length：卷积核的空域或时域长度

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”，“same”或“full”，full需要以theano为后端

* subsample_length：输出对输入的下采样因子

* atrous_rate:卷积核膨胀的系数，在其他地方也被称为'filter_dilation'

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
# apply an atrous convolution 1d with atrous rate 2 of length 3 to a sequence with 10 timesteps,
# with 64 output filters
model = Sequential()
model.add(AtrousConvolution1D(64, 3, atrous_rate=2, border_mode='same', input_shape=(10, 32)))
# now model.output_shape == (None, 10, 64)

# add a new atrous conv1d on top
model.add(AtrousConvolution1D(32, 3, atrous_rate=2, border_mode='same'))
# now model.output_shape == (None, 10, 32)
```

***
## Convolution2D层
```python
keras.layers.convolutional.Convolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```
二维卷积层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供```input_shape```参数。例如```input_shape = (3,128,128)```代表128*128的彩色RGB图像

### 参数

* nb_filter：卷积核的数目

* nb_row：卷积核的行数

* nb_col：卷积核的列数

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”，“same”或“full”，full需要以theano为后端

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
# apply a 3x3 convolution with 64 output filters on a 256x256 image:
model = Sequential()
model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
# now model.output_shape == (None, 64, 256, 256)

# add a 3x3 convolution on top, with 32 output filters:
model.add(Convolution2D(32, 3, 3, border_mode='same'))
# now model.output_shape == (None, 32, 256, 256)
```

***

## AtrousConvolution2D层
```python
keras.layers.convolutional.AtrousConvolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), atrous_rate=(1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```
该层对二维输入进行Atrous卷积，也即膨胀卷积或带孔洞的卷积。当使用该层作为第一层时，应提供```input_shape```参数。例如```input_shape = (3,128,128)```代表128*128的彩色RGB图像

### 参数

* nb_filter：卷积核的数目

* nb_row：卷积核的行数

* nb_col：卷积核的列数

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”，“same”，或“full”，full需要以theano为后端

* subsample：长为2的tuple，输出对输入的下采样因子，更普遍的称呼是“strides”

* atrous_rate：长为2的tuple，代表卷积核膨胀的系数，在其他地方也被称为'filter_dilation'

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

输出的行列数可能会因为填充而改变

### 例子

```python
# apply a 3x3 convolution with atrous rate 2x2 and 64 output filters on a 256x256 image:
model = Sequential()
model.add(AtrousConvolution2D(64, 3, 3, atrous_rate=(2,2), border_mode='valid', input_shape=(3, 256, 256)))
# now the actual kernel size is dilated from 3x3 to 5x5 (3+(3-1)*(2-1)=5)
# thus model.output_shape == (None, 64, 252, 252)
```

### 参考文献

* [<font color='#FF0000'>Multi-Scale Context Aggregation by Dilated Convolutions</font>](https://arxiv.org/abs/1511.07122)

***

## SeparableConvolution2D层
```python
keras.layers.convolutional.SeparableConvolution2D(nb_filter, nb_row, nb_col, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), depth_multiplier=1, dim_ordering='default', depthwise_regularizer=None, pointwise_regularizer=None, b_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, b_constraint=None, bias=True)
```
该层是对2D输入的可分离卷积

可分离卷积首先按深度方向进行卷积（对每个输入通道分别卷积），然后逐点进行卷积，将上一步的卷积结果混合到输出通道中。参数```depth_multiplier```控制了在depthwise卷积（第一步）的过程中，每个输入通道信号产生多少个输出通道。

直观来说，可分离卷积可以看做讲一个卷积核分解为两个小的卷积核，或看作Inception模块的一种极端情况。

当使用该层作为第一层时，应提供```input_shape```参数。例如```input_shape = (3,128,128)```代表128*128的彩色RGB图像

### Theano警告

该层目前只能在Tensorflow后端的条件下使用

### 参数

* nb_filter：卷积核的数目

* nb_row：卷积核的行数

* nb_col：卷积核的列数

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”，“same”，或“full”，full需要以theano为后端

* subsample：长为2的tuple，输出对输入的下采样因子，更普遍的称呼是“strides”

* depth_multiplier：在按深度卷积的步骤中，每个输入通道使用多少个输出通道

* depthwise_regularizer：施加在按深度卷积的权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* pointwise_regularizer：施加在按点卷积的权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[<font color='FF0000'>ActivityRegularizer</font>](../other/regularizers)对象

* depthwise_constraint：施加在按深度卷积权重上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* pointwise_constraint施加在按点卷积权重的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

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

***

## Deconvolution2D层
```python
keras.layers.convolutional.Deconvolution2D(nb_filter, nb_row, nb_col, output_shape, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1), dim_ordering='tf', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```
该层是卷积操作的转置（反卷积）。需要反卷积的情况通常发生在用户想要对一个普通卷积的结果做反方向的变换。例如，将具有该卷积层输出shape的tensor转换为具有该卷积层输入shape的tensor。，同时保留与卷积层兼容的连接模式。

当使用该层作为第一层时，应提供```input_shape```参数。例如```input_shape = (3,128,128)```代表128*128的彩色RGB图像

### 参数

* nb_filter：卷积核的数目

* nb_row：卷积核的行数

* nb_col：卷积核的列数

* output_shape：反卷积的输出shape，为整数的tuple，形如（nb_samples,nb_filter,nb_output_rows,nb_output_cols），计算output_shape的公式是：o = s (i - 1) + a + k - 2p,其中a的取值范围是0~s-1，其中：
	* i:输入的size（rows或cols）
	* k：卷积核大小（nb_filter）
	* s: 步长（subsample）
	* a：用户指定的的用于区别s个不同的可能output size的参数

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”，“same”，或“full”，full需要以theano为后端

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
# apply a 3x3 transposed convolution with stride 1x1 and 3 output filters on a 12x12 image:
model = Sequential()
model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 14, 14), border_mode='valid', input_shape=(3, 12, 12)))
# output_shape will be (None, 3, 14, 14)

# apply a 3x3 transposed convolution with stride 2x2 and 3 output filters on a 12x12 image:
model = Sequential()
model.add(Deconvolution2D(3, 3, 3, output_shape=(None, 3, 25, 25), subsample=(2, 2), border_mode='valid', input_shape=(3, 12, 12)))
model.summary()
# output_shape will be (None, 3, 25, 25)
```

### 参考文献
* [<font color='#FF0000'>A guide to convolution arithmetic for deep learning</font>](https://arxiv.org/abs/1603.07285)
* [<font color='#FF0000'>Transposed convolution arithmetic </font>](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic)
* [<font color='#FF0000'>Deconvolutional Networks </font>](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

***

## Convolution3D层
```python
keras.layers.convolutional.Convolution3D(nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```
三维卷积对三维的输入进行滑动窗卷积，当使用该层作为第一层时，应提供```input_shape```参数。例如```input_shape = (3,10,128,128)```代表对10帧128*128的彩色RGB图像进行卷积

### 参数

* nb_filter：卷积核的数目

* kernel_dim1：卷积核第1维度的长

* kernel_dim2：卷积核第2维度的长

* kernel_dim3：卷积核第3维度的长

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”，“same”，或“full”，full需要以theano为后端

* subsample：长为3的tuple，输出对输入的下采样因子，更普遍的称呼是“strides”
	
	*注意，subsample通过对3D卷积的结果以strides=（1，1，1）切片实现

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[<font color='FF0000'>ActivityRegularizer</font>](../other/regularizers)对象

* W_constraints：施加在权重上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* b_constraints：施加在偏置上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

* bias：布尔值，是否包含偏置向量（即层对输入做线性变换还是仿射变换）

### 输入shape

‘th’模式下，输入应为形如（samples，channels，input_dim1，input_dim2, input_dim3）的5D张量

‘tf’模式下，输入应为形如（samples，input_dim1，input_dim2, input_dim3，channels）的5D张量

这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的```input_shape```。

***

## Cropping1D层
```python
keras.layers.convolutional.Cropping1D(cropping=(1, 1))
```
在时间轴（axis1）上对1D输入（即时间序列）进行裁剪

### 参数

* cropping：长为2的tuple，指定在序列的首尾要裁剪掉多少个元素

### 输入shape

* 形如（samples，axis_to_crop，features）的3D张量

### 输出shape

* 形如（samples，cropped_axis，features）的3D张量

***
## Cropping2D层
```python
keras.layers.convolutional.Cropping2D(cropping=((0, 0), (0, 0)), dim_ordering='default')
```
对2D输入（图像）进行裁剪，将在空域维度，即宽和高的方向上裁剪

### 参数

* cropping：长为2的整数tuple，分别为宽和高方向上头部与尾部需要裁剪掉的元素数

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

### 输入shape

形如（samples，depth, first_axis_to_crop, second_axis_to_crop）


### 输出shape

形如(samples, depth, first_cropped_axis, second_cropped_axis)的4D张量

***
## Cropping3D层
```python
keras.layers.convolutional.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), dim_ordering='default')
```
对2D输入（图像）进行裁剪

### 参数

* cropping：长为3的整数tuple，分别为三个方向上头部与尾部需要裁剪掉的元素数

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

### 输入shape

形如 (samples, depth, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)的5D张量

### 输出shape
形如(samples, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)的5D张量
***
## UpSampling1D层
```python
keras.layers.convolutional.UpSampling1D(length=2)
```
在时间轴上，将每个时间步重复```length```次

### 参数

* length：上采样因子

### 输入shape

* 形如（samples，steps，features）的3D张量

### 输出shape

* 形如（samples，upsampled_steps，features）的3D张量

***

## UpSampling2D层
```python
keras.layers.convolutional.UpSampling2D(size=(2, 2), dim_ordering='th')
```
将数据的行和列分别重复size\[0\]和size\[1\]次

### 参数

* size：整数tuple，分别为行和列上采样因子

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

### 输入shape

‘th’模式下，为形如（samples，channels, rows，cols）的4D张量

‘tf’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

‘th’模式下，为形如（samples，channels, upsampled_rows, upsampled_cols）的4D张量

‘tf’模式下，为形如（samples，upsampled_rows, upsampled_cols，channels）的4D张量

***

## UpSampling3D层
```python
keras.layers.convolutional.UpSampling3D(size=(2, 2, 2), dim_ordering='th')
```
将数据的三个维度上分别重复size\[0\]、size\[1\]和ize\[2\]次

本层目前只能在使用Theano为后端时可用

### 参数

* size：长为3的整数tuple，代表在三个维度上的上采样因子

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

### 输入shape

‘th’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘tf’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

### 输出shape

‘th’模式下，为形如（samples, channels, dim1, dim2, dim3）的5D张量

‘tf’模式下，为形如（samples, upsampled_dim1, upsampled_dim2, upsampled_dim3,channels,）的5D张量

***

## ZeroPadding1D层
```python
keras.layers.convolutional.ZeroPadding1D(padding=1)
```
对1D输入的首尾端（如时域序列）填充0，以控制卷积以后向量的长度

### 参数

* padding：整数，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴1（第1维，第0维是样本数）

### 输入shape

形如（samples，axis_to_pad，features）的3D张量

### 输出shape

形如（samples，paded_axis，features）的3D张量

***

## ZeroPadding2D层
```python
keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')
```
对2D输入（如图片）的边界填充0，以控制卷积以后特征图的大小

### 参数

* padding：整数tuple，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴3和轴4（即在'th'模式下图像的行和列，在‘tf’模式下要填充的则是轴2，3）

dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

### 输入shape

‘th’模式下，形如（samples，channels，first_axis_to_pad，second_axis_to_pad）的4D张量

‘tf’模式下，形如（samples，first_axis_to_pad，second_axis_to_pad, channels）的4D张量

### 输出shape

‘th’模式下，形如（samples，channels，first_paded_axis，second_paded_axis）的4D张量

‘tf’模式下，形如（samples，first_paded_axis，second_paded_axis, channels）的4D张量

***

## ZeroPadding3D层
```python
keras.layers.convolutional.ZeroPadding3D(padding=(1, 1, 1), dim_ordering='th')
```
将数据的三个维度上填充0

本层目前只能在使用Theano为后端时可用

### 参数

padding：整数tuple，表示在要填充的轴的起始和结束处填充0的数目，这里要填充的轴是轴3，轴4和轴5，‘tf’模式下则是轴2，3和4

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

### 输入shape

‘th’模式下，为形如（samples, channels, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad,）的5D张量

‘tf’模式下，为形如（samples, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad, channels）的5D张量

### 输出shape

‘th’模式下，为形如（samples, channels, first_paded_axis，second_paded_axis, third_paded_axis,）的5D张量

‘tf’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量
