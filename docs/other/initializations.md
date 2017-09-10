# 初始化方法

初始化方法定义了对Keras层设置初始化权重的方法

不同的层可能使用不同的关键字来传递初始化方法，一般来说指定初始化方法的关键字是```kernel_initializer``` 和 ```bias_initializer```，例如：
```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

一个初始化器可以由字符串指定（必须是下面的预定义初始化器之一），或一个callable的函数，例如
```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```

## Initializer

Initializer是所有初始化方法的父类，不能直接使用，如果想要定义自己的初始化方法，请继承此类。

## 预定义初始化方法

### Zeros
```python
keras.initializers.Zeros()
```
全零初始化

### Ones
```python
keras.initializers.Ones()
```
全1初始化

### Constant
```python
keras.initializers.Constant(value=0)
```
初始化为固定值value

### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))
```
正态分布初始化

* mean：均值
* stddev：标准差
* seed：随机数种子

### RandomUniform
```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```
均匀分布初始化
* minval：均匀分布下边界
* maxval：均匀分布上边界
* seed：随机数种子


### TruncatedNormal
```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```
截尾高斯分布初始化，该初始化方法与RandomNormal类似，但位于均值两个标准差以外的数据将会被丢弃并重新生成，形成截尾分布。该分布是神经网络权重和滤波器的推荐初始化方法。

* mean：均值
* stddev：标准差
* seed：随机数种子

### VarianceScaling
```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```


该初始化方法能够自适应目标张量的shape。

当```distribution="normal"```时，样本从0均值，标准差为sqrt(scale / n)的截尾正态分布中产生。其中：

	* 当```mode = "fan_in"```时，权重张量的输入单元数。
	* 当```mode = "fan_out"```时，权重张量的输出单元数
	* 当```mode = "fan_avg"```时，权重张量的输入输出单元数的均值

当```distribution="uniform"```时，权重从[-limit, limit]范围内均匀采样，其中limit = limit = sqrt(3 * scale / n)

* scale: 放缩因子，正浮点数
* mode: 字符串，“fan_in”，“fan_out”或“fan_avg”fan_in", "fan_out", "fan_avg".
* distribution: 字符串，“normal”或“uniform”.
* seed: 随机数种子

### Orthogonal
```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

用随机正交矩阵初始化

* gain: 正交矩阵的乘性系数
* seed：随机数种子

参考文献：[Saxe et al.](http://arxiv.org/abs/1312.6120)

### Identiy
```python
keras.initializers.Identity(gain=1.0)
```
使用单位矩阵初始化，仅适用于2D方阵

* gain：单位矩阵的乘性系数

### lecun_uniform
```python
lecun_uniform(seed=None)
```

LeCun均匀分布初始化方法，参数由[-limit, limit]的区间中均匀采样获得，其中limit=sqrt(3 / fan_in), fin_in是权重向量的输入单元数（扇入）

* seed：随机数种子

参考文献：[LeCun 98, Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

### lecun_normal
```python
lecun_normal(seed=None)
```
LeCun正态分布初始化方法，参数由0均值，标准差为stddev = sqrt(1 / fan_in)的正态分布产生，其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目）

* seed：随机数种子

参考文献：

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
[Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)



### glorot_normal
```python
glorot_normal(seed=None)
```

Glorot正态分布初始化方法，也称作Xavier正态分布初始化，参数由0均值，标准差为sqrt(2 / (fan_in + fan_out))的正态分布产生，其中fan_in和fan_out是权重张量的扇入扇出（即输入和输出单元数目）

* seed：随机数种子

参考文献：[Glorot & Bengio, AISTATS 2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

###glorot_uniform

```python
glorot_uniform(seed=None)
```
Glorot均匀分布初始化方法，又成Xavier均匀初始化，参数从[-limit, limit]的均匀分布产生，其中limit为`sqrt(6 / (fan_in + fan_out))`。fan_in为权值张量的输入单元数，fan_out是权重张量的输出单元数。

* seed：随机数种子

参考文献：[Glorot & Bengio, AISTATS 2010](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

### he_normal
```python
he_normal(seed=None)
```

He正态分布初始化方法，参数由0均值，标准差为sqrt(2 / fan_in) 的正态分布产生，其中fan_in权重张量的扇入

* seed：随机数种子

参考文献：[He et al](http://arxiv.org/abs/1502.01852)


### he_uniform
```python
he_normal(seed=None)
```

LeCun均匀分布初始化方法，参数由[-limit, limit]的区间中均匀采样获得，其中limit=sqrt(6 / fan_in), fin_in是权重向量的输入单元数（扇入）

* seed：随机数种子

参考文献：[He et al](http://arxiv.org/abs/1502.01852)

## 自定义初始化器
如果需要传递自定义的初始化器，则该初始化器必须是callable的，并且接收```shape```（将被初始化的张量shape）和```dtype```（数据类型）两个参数，并返回符合```shape```和```dtype```的张量。


```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, init=my_init))
```