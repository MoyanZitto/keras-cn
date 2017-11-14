# 循环层Recurrent

## Recurrent层
```python
keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
```

这是循环层的抽象类，请不要在模型中直接应用该层（因为它是抽象类，无法实例化任何对象）。请使用它的子类```LSTM```，```GRU```或```SimpleRNN```。

所有的循环层（```LSTM```,```GRU```,```SimpleRNN```）都继承本层，因此下面的参数可以在任何循环层中使用。

### 参数

* weights：numpy array的list，用以初始化权重。该list形如```[(input_dim, output_dim),(output_dim, output_dim),(output_dim,)]```

* return_sequences：布尔值，默认```False```，控制返回类型。若为```True```则返回整个序列，否则仅返回输出序列的最后一个输出

* go_backwards：布尔值，默认为```False```，若为```True```，则逆向处理输入序列并返回逆序后的序列

* stateful：布尔值，默认为```False```，若为```True```，则一个batch中下标为i的样本的最终状态将会用作下一个batch同样下标的样本的初始状态。

* unroll：布尔值，默认为```False```，若为```True```，则循环层将被展开，否则就使用符号化的循环。当使用TensorFlow为后端时，循环网络本来就是展开的，因此该层不做任何事情。层展开会占用更多的内存，但会加速RNN的运算。层展开只适用于短序列。

* implementation：0，1或2， 若为0，则RNN将以更少但是更大的矩阵乘法实现，因此在CPU上运行更快，但消耗更多的内存。如果设为1，则RNN将以更多但更小的矩阵乘法实现，因此在CPU上运行更慢，在GPU上运行更快，并且消耗更少的内存。如果设为2（仅LSTM和GRU可以设为2），则RNN将把输入门、遗忘门和输出门合并为单个矩阵，以获得更加在GPU上更加高效的实现。注意，RNN dropout必须在所有门上共享，并导致正则效果性能微弱降低。

* input_dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape)

* input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接```Flatten```层，然后又要连接```Dense```层时，需要指定该参数，否则全连接的输出无法计算出来。注意，如果循环层不是网络的第一层，你需要在网络的第一层中指定序列的长度（通过```input_shape```指定）。

### 输入shape

形如（samples，timesteps，input_dim）的3D张量

### 输出shape

如果```return_sequences=True```：返回形如（samples，timesteps，output_dim）的3D张量

否则，返回形如（samples，output_dim）的2D张量

### 例子
```python
# as the first layer in a Sequential model
model = Sequential()
model.add(LSTM(32, input_shape=(10, 64)))
# now model.output_shape == (None, 32)
# note: `None` is the batch dimension.

# the following is identical:
model = Sequential()
model.add(LSTM(32, input_dim=64, input_length=10))

# for subsequent layers, no need to specify the input size:
         model.add(LSTM(16))

# to stack recurrent layers, you must use return_sequences=True
# on any recurrent layer that feeds into another recurrent layer.
# note that you only need to specify the input size on the first layer.
model = Sequential()
model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(10))
```

### 指定RNN初始状态的注意事项

可以通过设置`initial_state`用符号式的方式指定RNN层的初始状态。即，`initial_stat`的值应该为一个tensor或一个tensor列表，代表RNN层的初始状态。

也可以通过设置`reset_states`参数用数值的方法设置RNN的初始状态，状态的值应该为numpy数组或numpy数组的列表，代表RNN层的初始状态。

### 屏蔽输入数据（Masking）

循环层支持通过时间步变量对输入数据进行Masking，如果想将输入数据的一部分屏蔽掉，请使用[Embedding](embedding_layer)层并将参数```mask_zero```设为```True```。


### 使用状态RNN的注意事项

可以将RNN设置为‘stateful’，意味着由每个batch计算出的状态都会被重用于初始化下一个batch的初始状态。状态RNN假设连续的两个batch之中，相同下标的元素有一一映射关系。

要启用状态RNN，请在实例化层对象时指定参数```stateful=True```，并在Sequential模型使用固定大小的batch：通过在模型的第一层传入```batch_size=(...)```和```input_shape```来实现。在函数式模型中，对所有的输入都要指定相同的```batch_size```。

如果要将循环层的状态重置，请调用```.reset_states()```，对模型调用将重置模型中所有状态RNN的状态。对单个层调用则只重置该层的状态。


***

## SimpleRNN层
```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```
全连接RNN网络，RNN的输出会被回馈到输入

### 参数

* units：输出维度

* activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)）

* use_bias: 布尔值，是否使用偏置项

* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象

* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象

* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象

* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象

* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象

* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象

* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例

* 其他参数参考Recurrent的说明

### 参考文献

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

***

## GRU层
```python
keras.layers.recurrent.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```
门限循环单元（详见参考文献）

### 参数

* units：输出维度

* activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)）

* use_bias: 布尔值，是否使用偏置项

* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象

* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象

* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象

* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象

* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象

* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象

* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例

* 其他参数参考Recurrent的说明

### 参考文献

* [On the Properties of Neural Machine Translation: Encoder–Decoder Approaches](http://www.aclweb.org/anthology/W14-4012)

* [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/pdf/1412.3555v1.pdf)

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

***

## LSTM层
```python
keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```
Keras长短期记忆模型，关于此算法的详情，请参考[本教程](http://deeplearning.net/tutorial/lstm.html)

### 参数

* units：输出维度

* activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)）

* recurrent_activation: 为循环步施加的激活函数（参考[激活函数](../other/activations)）

* use_bias: 布尔值，是否使用偏置项

* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象

* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象

* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象

* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象

* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象

* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象

* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例

* 其他参数参考Recurrent的说明

### 参考文献

* [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)（original 1997 paper）

* [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)

* [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)



## ConvLSTM2D层
```python
keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
```
ConvLSTM2D是一个LSTM网络，但它的输入变换和循环变换是通过卷积实现的

### 参数

* filters: 整数，输出的维度，该参数含义同普通卷积层的filters
* kernel_size: 整数或含有n个整数的tuple/list，指定卷积窗口的大小
* strides: 整数或含有n个整数的tuple/list，指定卷积步长，当不等于1时，无法使用dilation功能，即dialation_rate必须为1.
* padding: "valid" 或 "same" 之一
* data_format: * data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是```~/.keras/keras.json```中设置的值，若从未设置过，则为“channels_last”。
* dilation_rate: 单个整数或由两个个整数构成的list/tuple，指定dilated convolution中的膨胀比例。任何不为1的dilation_rate均与任何不为1的strides均不兼容。
* activation: activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
* recurrent_activation: 用在recurrent部分的激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)），或逐元素（element-wise）的Theano函数。如果不指定该参数，将不会使用任何激活函数（即使用线性激活函数：a(x)=x）
* use_bias: Boolean, whether the layer uses a bias vector.
* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)
* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)
* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)
* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象
* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象
* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象
* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象
* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象
* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象
* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象
* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例
* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
* 其他参数参考Recurrent的说明

### 输入shape
若data_format='channels_first'， 为形如(samples,time, channels, rows, cols)的5D tensor
若data_format='channels_last' 为形如(samples,time, rows, cols, channels)的5D tensor

### 输出shape

if return_sequences：
	if data_format='channels_first' ：5D tensor (samples, time, filters, output_row, output_col)
	if data_format='channels_last'  ：5D tensor (samples, time, output_row, output_col, filters)
else
	if data_format ='channels_first' :4D tensor (samples, filters, output_row, output_col)
	if data_format='channels_last'   :4D tensor  (samples, output_row, output_col, filters) (o_row和o_col由filter和padding决定)

### 参考文献

[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1) 
* 当前的实现不包含cell输出上的反馈循环（feedback loop）


## SimpleRNNCell层
```python
keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```
SinpleRNN的Cell类

### 参数

* units：输出维度

* activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)）

* use_bias: 布尔值，是否使用偏置项

* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象

* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象

* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象

* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象

* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象

* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象

* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例


## GRUCell层
```python
keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```
GRU的Cell类

### 参数

* units：输出维度

* activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)）

* use_bias: 布尔值，是否使用偏置项

* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象

* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象

* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象

* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象

* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象

* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象

* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
* 其他参数参考Recurrent的说明


## LSTMCell层
```python
keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```
LSTM的Cell类

### 参数

* units：输出维度

* activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)）

* use_bias: 布尔值，是否使用偏置项

* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象

* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象

* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象

* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象

* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象

* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象

* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
* 其他参数参考Recurrent的说明


## StackedRNNCells层
```python
keras.layers.StackedRNNCells(cells)
```
这是一个wrapper，用于将多个recurrent cell包装起来，使其行为类型单个cell。该层用于实现搞笑的stacked RNN

### 参数

* cells：list，其中每个元素都是一个cell对象

### 例子
```python
cells = [
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
    keras.layers.LSTMCell(output_dim),
]

inputs = keras.Input((timesteps, input_dim))
x = keras.layers.StackedRNNCells(cells)(inputs)
```



## CuDNNGRU层
```python
keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

基于CuDNN的快速GRU实现，只能在GPU上运行，只能使用tensoflow为后端

### 参数
* units：输出维度

* activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)）

* use_bias: 布尔值，是否使用偏置项

* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象

* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象

* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象

* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象

* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象

* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象

* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
* 其他参数参考Recurrent的说明



## CuDNNLSTM层
```python
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

基于CuDNN的快速LSTM实现，只能在GPU上运行，只能使用tensoflow为后端

### 参数
* units：输出维度

* activation：激活函数，为预定义的激活函数名（参考[激活函数](../other/activations)）

* use_bias: 布尔值，是否使用偏置项

* kernel_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* recurrent_initializer：循环核的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* bias_initializer：权值初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* kernel_regularizer：施加在权重上的正则项，为[Regularizer](../other/regularizers)对象

* bias_regularizer：施加在偏置向量上的正则项，为[Regularizer](../other/regularizers)对象

* recurrent_regularizer：施加在循环核上的正则项，为[Regularizer](../other/regularizers)对象

* activity_regularizer：施加在输出上的正则项，为[Regularizer](../other/regularizers)对象

* kernel_constraints：施加在权重上的约束项，为[Constraints](../other/constraints)对象

* recurrent_constraints：施加在循环核上的约束项，为[Constraints](../other/constraints)对象

* bias_constraints：施加在偏置上的约束项，为[Constraints](../other/constraints)对象

* dropout：0~1之间的浮点数，控制输入线性变换的神经元断开比例

* recurrent_dropout：0~1之间的浮点数，控制循环状态的线性变换的神经元断开比例
* 其他参数参考Recurrent的说明