# 噪声层Noise

## GaussianNoise层
```python
keras.layers.noise.GaussianNoise(stddev)
```

为数据施加0均值，标准差为```stddev```的加性高斯噪声。该层在克服过拟合时比较有用，你可以将它看作是随机的数据提升。高斯噪声是需要对输入数据进行破坏时的自然选择。


因为这是一个起正则化作用的层，该层只在训练时才有效。

### 参数

* stddev：浮点数，代表要产生的高斯噪声标准差

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

***

## GaussianDropout层
```python
keras.layers.noise.GaussianDropout(rate)
```
为层的输入施加以1为均值，标准差为```sqrt(rate/(1-rate)```的乘性高斯噪声

因为这是一个起正则化作用的层，该层只在训练时才有效。

### 参数

* rate：浮点数，断连概率，与[Dropout层](core_layer/#dropout)相同

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

### 参考文献

* [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)



## AlphaDropout
```python
keras.layers.noise.AlphaDropout(rate, noise_shape=None, seed=None)
```
对输入施加Alpha Dropout

Alpha Dropout是一种保持输入均值和方差不变的Dropout，该层的作用是即使在dropout时也保持数据的自规范性。 通过随机对负的饱和值进行激活，Alphe Drpout与selu激活函数配合较好。


### 参数

* rate: 浮点数，类似Dropout的Drop比例。乘性mask的标准差将保证为`sqrt(rate / (1 - rate))`.
* seed: 随机数种子

### 输入shape

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出shape

与输入相同

### 参考文献

[Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)