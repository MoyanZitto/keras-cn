# 嵌入层 Embedding

## Embedding层

```python
keras.layers.embeddings.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```
嵌入层将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]

Embedding层只能作为模型的第一层

### 参数

* input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1

* output_dim：大于0的整数，代表全连接嵌入的维度

* embeddings_initializer: 嵌入矩阵的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的初始化器。参考[initializers](../other/initializations)

* embeddings_regularizer: 嵌入矩阵的正则项，为[Regularizer](../other/regularizers)对象

* embeddings_constraint: 嵌入矩阵的约束项，为[Constraints](../other/constraints)对象

* mask_zero：布尔值，确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值，该参数在使用[递归层](recurrent_layer)处理变长输入时有用。设置为```True```的话，模型中后续的层必须都支持masking，否则会抛出异常。如果该值为True，则下标0在字典中不可用，input_dim应设置为|vocabulary| + 1。 

* input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接```Flatten```层，然后接```Dense```层，则必须指定该参数，否则```Dense```层的输出维度无法自动推断。


### 输入shape

形如（samples，sequence_length）的2D张量

### 输出shape

形如(samples, sequence_length, output_dim)的3D张量

### 例子
```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

### 参考文献

* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)