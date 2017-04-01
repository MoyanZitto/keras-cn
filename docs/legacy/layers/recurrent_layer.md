# 递归层Recurrent

## Recurrent层
```python
keras.layers.recurrent.Recurrent(weights=None, return_sequences=False, go_backwards=False, stateful=False, unroll=False, consume_less='cpu', input_dim=None, input_length=None)
```

这是递归层的抽象类，请不要在模型中直接应用该层（因为它是抽象类，无法实例化任何对象）。请使用它的子类```LSTM```或```SimpleRNN```。

所有的递归层（```LSTM```,```GRU```,```SimpleRNN```）都服从本层的性质，并接受本层指定的所有关键字参数。

### 参数

* weights：numpy array的list，用以初始化权重。该list形如```[(input_dim, output_dim),(output_dim, output_dim),(output_dim,)]```

* return_sequences：布尔值，默认```False```，控制返回类型。若为```True```则返回整个序列，否则仅返回输出序列的最后一个输出

* go_backwards：布尔值，默认为```False```，若为```True```，则逆向处理输入序列

* stateful：布尔值，默认为```False```，若为```True```，则一个batch中下标为i的样本的最终状态将会用作下一个batch同样下标的样本的初始状态。

* unroll：布尔值，默认为```False```，若为```True```，则递归层将被展开，否则就使用符号化的循环。当使用TensorFlow为后端时，递归网络本来就是展开的，因此该层不做任何事情。层展开会占用更多的内存，但会加速RNN的运算。层展开只适用于短序列。

* consume_less：‘cpu’或‘mem’之一。若设为‘cpu’，则RNN将使用较少、较大的矩阵乘法来实现，从而在CPU上会运行更快，但会更消耗内存。如果设为‘mem’，则RNN将会较多的小矩阵乘法来实现，从而在GPU并行计算时会运行更快（但在CPU上慢），并占用较少内存。

* input_dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape)

* input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接```Flatten```层，然后又要连接```Dense```层时，需要指定该参数，否则全连接的输出无法计算出来。注意，如果递归层不是网络的第一层，你需要在网络的第一层中指定序列的长度，如通过```input_shape```指定。

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
# now model.output_shape == (None, 10, 32)
# note: `None` is the batch dimension.

# the following is identical:
model = Sequential()
model.add(LSTM(32, input_dim=64, input_length=10))

# for subsequent layers, not need to specify the input size:
model.add(LSTM(16))
```

### 屏蔽输入数据（Masking）

递归层支持通过时间步变量对输入数据进行Masking，如果想将输入数据的一部分屏蔽掉，请使用[<font color-'#FF0000'>Embedding</font>](embedding_layer)层并将参数```mask_zero```设为```True```。

### TensorFlow警告

目前为止，当使用TensorFlow作为后端时，序列的时间步数目必须在网络中指定。通过```input_length```（如果网络首层是递归层）或完整的```input_shape```来指定该值。

### 使用状态RNN的注意事项

可以将RNN设置为‘stateful’，意味着训练时每个batch的状态都会被重用于初始化下一个batch的初始状态。状态RNN假设连续的两个batch之中，相同下标的元素有一一映射关系。

要启用状态RNN，请在实例化层对象时指定参数```stateful=True```，并指定模型使用固定大小的batch：通过在模型的第一层传入```batch_input_shape=(...)```来实现。该参数应为包含batch大小的元组，例如（32，10，100）代表每个batch的大小是32.

如果要将递归层的状态重置，请调用```.reset_states()```，对模型调用将重置模型中所有状态RNN的状态。对单个层调用则只重置该层的状态。

### 以TensorFlow作为后端时使用dropout的注意事项

当使用TensorFlow作为后端时，如果要在递归层使用dropout，需要同上面所述的一样指定好固定的batch大小

***

## SimpleRNN层
```python
keras.layers.recurrent.SimpleRNN(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
```
全连接RNN网络，RNN的输出会被回馈到输入

### 参数

* output_dim：内部投影和输出的维度

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。

* inner_init：内部单元的初始化方法

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)）

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* U_regularizer：施加在递归权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* dropout_W：0~1之间的浮点数，控制输入单元到输入门的连接断开比例

* dropout_U：0~1之间的浮点数，控制输入单元到递归连接的断开比例

### 参考文献

* [<font color='FF0000'>A Theoretically Grounded Application of Dropout in Recurrent Neural Networks</font>](http://arxiv.org/abs/1512.05287)

***

## GRU层
```python
keras.layers.recurrent.GRU(output_dim, init='glorot_uniform', inner_init='orthogonal', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
```
门限递归单元（详见参考文献）

### 参数

* output_dim：内部投影和输出的维度

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。

* inner_init：内部单元的初始化方法

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)）

* inner_activation：内部单元激活函数

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* U_regularizer：施加在递归权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* dropout_W：0~1之间的浮点数，控制输入单元到输入门的连接断开比例

* dropout_U：0~1之间的浮点数，控制输入单元到递归连接的断开比例

### 参考文献

* [<font color='FF0000'>On the Properties of Neural Machine Translation: Encoder–Decoder Approaches</font>](http://www.aclweb.org/anthology/W14-4012)

* [<font color='FF0000'>Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling</font>](http://arxiv.org/pdf/1412.3555v1.pdf)

* [<font color='FF0000'>A Theoretically Grounded Application of Dropout in Recurrent Neural Networks</font>](http://arxiv.org/abs/1512.05287)

***

## LSTM层
```python
keras.layers.recurrent.LSTM(output_dim, init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
```
Keras长短期记忆模型，关于此算法的详情，请参考[<font color='FF0000'>本教程</font>](http://deeplearning.net/tutorial/lstm.html)

### 参数

* output_dim：内部投影和输出的维度

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。

* inner_init：内部单元的初始化方法

* forget_bias_init：遗忘门偏置的初始化函数，[<font color='FF0000'>Jozefowicz et al.</font>](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)建议初始化为全1元素

* activation：激活函数，为预定义的激活函数名（参考[<font color='#FF0000'>激活函数</font>](../other/activations)）

* inner_activation：内部单元激活函数

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* U_regularizer：施加在递归权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* b_regularizer：施加在偏置向量上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* dropout_W：0~1之间的浮点数，控制输入单元到输入门的连接断开比例

* dropout_U：0~1之间的浮点数，控制输入单元到递归连接的断开比例

### 参考文献

* [<font color='FF0000'>Long short-term memory </font>](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)（original 1997 paper）

* [<font color='FF0000'>Learning to forget: Continual prediction with LSTM</font>](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)

* [<font color='FF0000'>Supervised sequence labelling with recurrent neural networks</font>](http://www.cs.toronto.edu/~graves/preprint.pdf)

* [<font color='FF0000'>A Theoretically Grounded Application of Dropout in Recurrent Neural Networks</font>](http://arxiv.org/abs/1512.05287)
