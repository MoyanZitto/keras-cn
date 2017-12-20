#Merge层

Merge层提供了一系列用于融合两个层或两个张量的层对象和方法。以大写首字母开头的是Layer类，以小写字母开头的是张量的函数。小写字母开头的张量函数在内部实际上是调用了大写字母开头的层。

## Add
```python
keras.layers.Add()
```
添加输入列表的图层。

该层接收一个相同shape列表张量，并返回它们的和，shape不变。

### Example

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.Add()([x1, x2])  # equivalent to added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

## SubStract
```python
keras.layers.Subtract()
```
两个输入的层相减。

它将大小至少为2，相同Shape的列表张量作为输入，并返回一个张量（输入[0] - 输入[1]），也是相同的Shape。


### Example

```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# Equivalent to subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

## Multiply
```python
keras.layers.Multiply()
```
该层接收一个列表的同shape张量，并返回它们的逐元素积的张量，shape不变。

## Average
```python
keras.layers.Average()
```
该层接收一个列表的同shape张量，并返回它们的逐元素均值，shape不变。


## Maximum
```python
keras.layers.Maximum()
```
该层接收一个列表的同shape张量，并返回它们的逐元素最大值，shape不变。

## Concatenate
```python
keras.layers.Concatenate(axis=-1)
```
该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。

### 参数

* axis: 想接的轴
* **kwargs: 普通的Layer关键字参数

## Dot
```python
keras.layers.Dot(axes, normalize=False)
```
计算两个tensor中样本的张量乘积。例如，如果两个张量```a```和```b```的shape都为（batch_size, n），则输出为形如（batch_size,1）的张量，结果张量每个batch的数据都是a[i,:]和b[i,:]的矩阵（向量）点积。


### 参数

* axes: 整数或整数的tuple，执行乘法的轴。
* normalize: 布尔值，是否沿执行成绩的轴做L2规范化，如果设为True，那么乘积的输出是两个样本的余弦相似性。
* **kwargs: 普通的Layer关键字参数


## add
```python
keras.layers.add(inputs)
```
Add层的函数式包装

###参数：

* inputs: 长度至少为2的张量列表A
* **kwargs: 普通的Layer关键字参数
###返回值

输入列表张量之和

### Example 
```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```


## subtract
```python
keras.layers.subtract(inputs)
```
Subtract层的函数式包装

###参数：

* inputs: 长度至少为2的张量列表A
* **kwargs: 普通的Layer关键字参数
###返回值

输入张量列表的差别

### Example 
```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

## multiply
```python
keras.layers.multiply(inputs)
```
Multiply的函数式包装

###参数：

* inputs: 长度至少为2的张量列表
* **kwargs: 普通的Layer关键字参数
###返回值

输入列表张量之逐元素积

## average
```python
keras.layers.average(inputs)
```
Average的函数包装

###参数：

* inputs: 长度至少为2的张量列表
* **kwargs: 普通的Layer关键字参数
###返回值

输入列表张量之逐元素均值

## maximum
```python
keras.layers.maximum(inputs)
```
Maximum的函数包装

###参数：

* inputs: 长度至少为2的张量列表
* **kwargs: 普通的Layer关键字参数
###返回值

输入列表张量之逐元素均值


## concatenate
```python
keras.layers.concatenate(inputs, axis=-1)
```
Concatenate的函数包装

### 参数
* inputs: 长度至少为2的张量列
* axis: 相接的轴
* **kwargs: 普通的Layer关键字参数

## dot
```python
keras.layers.dot(inputs, axes, normalize=False)
```
Dot的函数包装


### 参数
* inputs: 长度至少为2的张量列
* axes: 整数或整数的tuple，执行乘法的轴。
* normalize: 布尔值，是否沿执行成绩的轴做L2规范化，如果设为True，那么乘积的输出是两个样本的余弦相似性。
* **kwargs: 普通的Layer关键字参数
