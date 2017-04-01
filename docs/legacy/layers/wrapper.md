# 包装器Wrapper

## TimeDistributed包装器
```python
keras.layers.wrappers.TimeDistributed(layer)
```
该包装器可以把一个层应用到输入的每一个时间步上

### 参数

* layer：Keras层对象

输入至少为3D张量，下标为1的维度将被认为是时间维

例如，考虑一个含有32个样本的batch，每个样本都是10个向量组成的序列，每个向量长为16，则其输入维度为```(32,10,16)```，其不包含batch大小的```input_shape```为```(10,16)```

我们可以使用包装器```TimeDistributed```包装```Dense```，以产生针对各个时间步信号的独立全连接：

```python
# as the first layer in a model
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)

# subsequent layers: no need for input_shape
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

程序的输出数据shape为```(32,10,8)```

使用```TimeDistributed```包装```Dense```严格等价于```layers.TimeDistribuedDense```。不同的是包装器```TimeDistribued```还可以对别的层进行包装，如这里对```Convolution2D```包装：

```python
model = Sequential()
model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(10, 3, 299, 299)))
```

## Bidirectional包装器
```python
keras.layers.wrappers.Bidirectional(layer, merge_mode='concat', weights=None)
```
双向RNN包装器

### 参数

* layer：```Recurrent```对象
* merge_mode：前向和后向RNN输出的结合方式，为```sum```,```mul```,```concat```,```ave```和```None```之一，若设为None，则返回值不结合，而是以列表的形式返回

### 例子
```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```