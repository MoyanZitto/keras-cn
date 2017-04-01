# 嵌入层 Embedding

## Embedding层

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, init='uniform', input_length=None, W_regularizer=None, activity_regularizer=None, W_constraint=None, mask_zero=False, weights=None, dropout=0.0)
```
嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]

Embedding层只能作为模型的第一层

### 参数

* input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1

* output_dim：大于0的整数，代表全连接嵌入的维度

* init：初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* weights：权值，为numpy array的list。该list应仅含有一个如（input_dim,output_dim）的权重矩阵

* W_regularizer：施加在权重上的正则项，为[<font color='FF0000'>WeightRegularizer</font>](../other/regularizers)对象

* W_constraints：施加在权重上的约束项，为[<font color='FF0000'>Constraints</font>](../other/constraints)对象

* mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用[<font color='#FF0000'>递归层</font>](recurrent_layer)处理变长输入时有用。设置为```True```的话，模型中后续的层必须都支持masking，否则会抛出异常

* input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接```Flatten```层，然后接```Dense```层，则必须指定该参数，否则```Dense```层的输出维度无法自动推断。

* dropout：0~1的浮点数，代表要断开的嵌入比例，

### 输入shape

形如（samples，sequence_length）的2D张量

### 输出shape

形如(samples, sequence_length, output_dim)的3D张量

### 参考文献

* [<font color='FF0000'>A Theoretically Grounded Application of Dropout in Recurrent Neural Networks</font>](http://arxiv.org/abs/1512.05287)