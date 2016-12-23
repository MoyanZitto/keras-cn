# 池化层

## MaxPooling1D层
```python
keras.layers.convolutional.MaxPooling1D(pool_length=2, stride=None, border_mode='valid')
```
对时域1D信号进行最大值池化

### 参数

* pool_length：下采样因子，如取2则将输入下采样到一半长度

* stride：整数或None，步长值

* border_mode：‘valid’或者‘same’
	
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


* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

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

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

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

dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

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

* dim_ordering：dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第4个位置。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。
### 输入shape

‘th’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘tf’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

### 输出shape

‘th’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量

‘tf’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

***

## GlobalMaxPooling1D层
```python
keras.layers.pooling.GlobalMaxPooling1D()
```
对于时间信号的全局最大池化

### 输入shape

* 形如（samples，steps，features）的3D张量

### 输出shape

* 形如(samples, features)的2D张量

***

## GlobalAveragePooling1D层
```python
keras.layers.pooling.GlobalAveragePooling1D()
```
为时域信号施加全局平均值池化

### 输入shape

* 形如（samples，steps，features）的3D张量

### 输出shape

* 形如(samples, features)的2D张量

***

## GlobalMaxPooling2D层
```python
keras.layers.pooling.GlobalMaxPooling2D(dim_ordering='default')
```
为空域信号施加全局最大值池化

### 参数

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

### 输入shape

‘th’模式下，为形如（samples，channels, rows，cols）的4D张量

‘tf’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

形如(nb_samples, channels)的2D张量

***

## GlobalAveragePooling2D层
```python
keras.layers.pooling.GlobalAveragePooling2D(dim_ordering='default')
```
为空域信号施加全局平均值池化

### 参数

* dim_ordering：‘th’或‘tf’。‘th’模式中通道维（如彩色图像的3通道）位于第1个位置（维度从0开始算），而在‘tf’模式中，通道维位于第3个位置。例如128*128的三通道彩色图片，在‘th’模式中```input_shape```应写为（3，128，128），而在‘tf’模式中应写为（128，128，3），注意这里3出现在第0个位置，因为```input_shape```不包含样本数的维度，在其内部实现中，实际上是（None，3，128，128）和（None，128，128，3）。默认是```image_dim_ordering```指定的模式，可在```~/.keras/keras.json```中查看，若没有设置过则为'tf'。

### 输入shape

‘th’模式下，为形如（samples，channels, rows，cols）的4D张量

‘tf’模式下，为形如（samples，rows, cols，channels）的4D张量

### 输出shape

形如(nb_samples, channels)的2D张量

