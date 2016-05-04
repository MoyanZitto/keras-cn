# 噪声层Noise

## GaussianNoise层
```python
keras.layers.noise.GaussianNoise(sigma)
```
为层的输入施加0均值，标准差为```sigma```的加性高斯噪声。该层在克服过拟合时比较有用，你可以将它看作是随机的数据提升。高斯噪声是需要对输入数据进行破坏时的自然选择。

因为这是一个起正则化作用的层，该层只在训练时才有效。

### 参数

* sigma：浮点数，代表要产生的高斯噪声标准差

### 输入形状

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出形状

与输入相同

***

## GaussianDropout层
```python
keras.layers.noise.GaussianDropout(p)
```
为层的输入施加以1为均值，标准差为```sqrt(p/(1-p)```的乘性高斯噪声

因为这是一个起正则化作用的层，该层只在训练时才有效。

### 参数

* p：浮点数，断连概率，与[<font color='#FF0000'>Dropout层</font>](core_layer/#dropout)相同

### 输入形状

任意，当使用该层为模型首层时需指定```input_shape```参数

### 输出形状

与输入相同

### 参考文献

* [<font color='FF0000'>Dropout: A Simple Way to Prevent Neural Networks from Overfitting</font>](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)