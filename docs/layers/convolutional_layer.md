# 卷积层

## Convolution1D层
```python
keras.layers.convolutional.Convolution1D(nb_filter, filter_length, init='uniform', activation='linear', weights=None, border_mode='valid', subsample_length=1, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None, input_length=None)
```

一维卷积层，用以在一维输入信号上进行领域滤波。当使用该层作为首层时，需要提供关键字参数```input_dim```或```input_shape```。例如```input_dim=128```长为128的向量序列输入，而```input_shape=(10,128)```代表一个长为10的128向量序列

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

* border_mode：边界模式，为“valid”或“same”

* subsample：长为2的tuple，输出对输入的下采样因子，更普遍的称呼是“strides”

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[<font color='FF0000'>ActivityRegularizer</font>](../other/regularizers)对象

* W_constraints：施加在权重上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* b_constraints：施加在偏置上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在低0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是‘th’模式。

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

## Convolution3D层
```python
keras.layers.convolutional.Convolution3D(nb_filter, kernel_dim1, kernel_dim2, kernel_dim3, init='glorot_uniform', activation='linear', weights=None, border_mode='valid', subsample=(1, 1, 1), dim_ordering='th', W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True)
```
三维卷积对三维的输入进行滑动窗卷积，当使用该层作为第一层时，应提供```input_shape```参数。例如```input_shape = (3,10,128,128)```代表对10帧128*128的彩色RGB图像进行卷积

目前，该层仅仅在使用Theano作为后端时可用

### 参数

* nb_filter：卷积核的数目

* kernel_dim1：卷积核第1维度的长

* kernel_dim2：卷积核第2维度的长

* kernel_dim3：卷积核第3维度的长

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）

* weights：权值，为numpy array的list。该list应含有一个形如（input_dim,output_dim）的权重矩阵和一个形如(output_dim,)的偏置向量。

* border_mode：边界模式，为“valid”或“same”

* subsample：长为3的tuple，输出对输入的下采样因子，更普遍的称呼是“strides”
	
	*注意，subsample通过对3D卷积的结果以strides=（1，1，1）切片实现

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[<font color='FF0000'>ActivityRegularizer</font>](../other/regularizers)对象

* W_constraints：施加在权重上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* b_constraints：施加在偏置上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

* bias：布尔值，是否包含偏置向量（即层对输入做线性变换还是仿射变换）

### 输入shape

‘th’模式下，输入应为形如（samples，channels，input_dim1，input_dim2, input_dim3）的5D张量

‘tf’模式下，输入应为形如（samples，input_dim1，input_dim2, input_dim3，channels）的5D张量

同样的，这里的输入shape指的是函数内部实现的输入shape，而非函数接口应指定的```input_shape```。

***

## MaxPooling1D层
```python
keras.layers.convolutional.MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
```
对时域1D信号进行最大值池化

### 参数

* pool_length：下采样因子，如取2则将输入下采样到一半长度

* stride：整数或None，步长值

* border_mode：‘valid’或者‘same’
	* 注意，目前‘same’模式只能在TensorFlow作为后端时使用
	
### 输入shape

* 形如（samples，steps，features）的3D张量

### 输出shape

* 形如（samples，downsampled_steps，features）的3D张量

***

## MaxPooling2D层
```python
keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th')
```
为空域信号施加最大值池化

### 参数

* pool_size：长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半

* strides：长为2的整数tuple，或者None，步长值。

* border_mode：‘valid’或者‘same’
	* 注意，目前‘same’模式只能在TensorFlow作为后端时使用

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

### 输入shape

‘th’模式下，为形如（samples，channels, rows，cols）的4D张量

‘tf’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

‘th’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量

‘tf’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量

***

## MaxPooling3D层
```python
keras.layers.convolutional.MaxPooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='th')
```
为3D信号（空域或时空域）施加最大值池化

本层目前只能在使用Theano为后端时可用

### 参数

* pool_size：长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。

* strides：长为3的整数tuple，或者None，步长值。

* border_mode：‘valid’或者‘same’

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

### 输入shape

‘th’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘tf’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

### 输出shape

‘th’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量

‘tf’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

***

## AveragePooling1D层
```python
keras.layers.convolutional.AveragePooling1D(pool_length=2, stride=None, border_mode='valid')
```
对时域1D信号进行平均值池化

### 参数

* pool_length：下采样因子，如取2则将输入下采样到一半长度

* stride：整数或None，步长值

* border_mode：‘valid’或者‘same’
	* 注意，目前‘same’模式只能在TensorFlow作为后端时使用
	
### 输入shape

* 形如（samples，steps，features）的3D张量

### 输出shape

* 形如（samples，downsampled_steps，features）的3D张量

***

## AveragePooling2D层
```python
keras.layers.convolutional.AveragePooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='th')
```
为空域信号施加平均值池化

### 参数

* pool_size：长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半

* strides：长为2的整数tuple，或者None，步长值。

* border_mode：‘valid’或者‘same’
	* 注意，目前‘same’模式只能在TensorFlow作为后端时使用

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

### 输入shape

‘th’模式下，为形如（samples，channels, rows，cols）的4D张量

‘tf’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

‘th’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量

‘tf’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量

***

## AveragePooling3D层
```python
keras.layers.convolutional.AveragePooling3D(pool_size=(2, 2, 2), strides=None, border_mode='valid', dim_ordering='th')
```
为3D信号（空域或时空域）施加平均值池化

本层目前只能在使用Theano为后端时可用

### 参数

* pool_size：长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。

* strides：长为3的整数tuple，或者None，步长值。

* border_mode：‘valid’或者‘same’

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

### 输入shape

‘th’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘tf’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

### 输出shape

‘th’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量

‘tf’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

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

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

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

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

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

dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

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

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置，```Convolution2D```有较详细的类似说明。默认是‘th’模式。

### 输入shape

‘th’模式下，为形如（samples, channels, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad,）的5D张量

‘tf’模式下，为形如（samples, first_axis_to_pad，first_axis_to_pad, first_axis_to_pad, channels）的5D张量

### 输出shape

‘th’模式下，为形如（samples, channels, first_paded_axis，second_paded_axis, third_paded_axis,）的5D张量

‘tf’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量