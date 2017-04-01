# 约束项

来自```constraints```模块的函数在优化过程中为网络的参数施加约束

惩罚项基于层进行惩罚，目前惩罚项的接口与层有关，但```Dense, TimeDistributedDense, MaxoutDense, Covolution1D, Covolution2D, Convolution3D```具有共同的接口。

这些层通过一下关键字施加约束项

* ```W_constraint```：对主权重矩阵进行约束

* ```b_constraint```：对偏置向量进行约束

```python
from keras.constraints import maxnorm
model.add(Dense(64, W_constraint = maxnorm(2)))
```

## 预定义约束项

* maxnorm(m=2)：最大模约束

* nonneg()：非负性约束

* unitnorm()：单位范数约束, 强制矩阵沿最后一个轴拥有单位范数