# 循环层Recurrent

## Recurrent层
```python
keras.layers.recurrent.Recurrent(return_sequences=False, go_backwards=False, stateful=False, unroll=False, implementation=0)
```

这是循环层的抽象类，请不要在模型中直接应用该层（因为它是抽象类，无法实例化任何对象）。请使用它的子类```LSTM```，```GRU```或```SimpleRNN```。

所有的循环层（```LSTM```,```GRU```,```SimpleRNN```）都服从本层的性质，并接受本层指定的所有关键字参数。

### 参数

* weights：numpy array的list，用以初始化权重。该list形如```[(input_dim, output_dim),(output_dim, output_dim),(output_dim,)]```

* return_sequences：布尔值，默认```False```，控制返回类型。若为```True```则返回整个序列，否则仅返回输出序列的最后一个输出

* go_backwards：布尔值，默认为```False```，若为```True```，则逆向处理输入序列

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

# for subsequent layers, not need to specify the input size:
model.add(LSTM(16))
```

### 屏蔽输入数据（Masking）

循环层支持通过时间步变量对输入数据进行Masking，如果想将输入数据的一部分屏蔽掉，请使用[Embedding](embedding_layer)层并将参数```mask_zero```设为```True```。


### 使用状态RNN的注意事项

可以将RNN设置为‘stateful’，意味着由每个batch计算出的状态都会被重用于初始化下一个batch的初始状态。状态RNN假设连续的两个batch之中，相同下标的元素有一一映射关系。

要启用状态RNN，请在实例化层对象时指定参数```stateful=True```，并在Sequential模型使用固定大小的batch：通过在模型的第一层传入```batch_size=(...)```和```input_shape```来实现。在函数式模型中，对所有的输入都要指定相同的```batch_size```。

如果要将循环层的状态重置，请调用```.reset_states()```，对模型调用将重置模型中所有状态RNN的状态。对单个层调用则只重置该层的状态。


***

## SimpleRNN层
```python
keras.layers.recurrent.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
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

### 参考文献

* [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)（original 1997 paper）

* [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)

* [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
