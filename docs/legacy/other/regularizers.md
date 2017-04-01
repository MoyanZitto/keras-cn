# 正则项

正则项在优化过程中层的参数或层的激活值添加惩罚项，这些惩罚项将与损失函数一起作为网络的最终优化目标

惩罚项基于层进行惩罚，目前惩罚项的接口与层有关，但```Dense, TimeDistributedDense, MaxoutDense, Covolution1D, Covolution2D, Convolution3D```具有共同的接口。

这些层有三个关键字参数以施加正则项：

* ```W_regularizer```：施加在权重上的正则项，为```WeightRegularizer```对象

* ```b_regularizer```：施加在偏置向量上的正则项，为```WeightRegularizer```对象

* ```activity_regularizer```：施加在输出上的正则项，为```ActivityRegularizer```对象

## 例子
```python
from keras.regularizers import l2, activity_l2
model.add(Dense(64, input_dim=64, W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
```

## 预定义正则项

```python
keras.regularizers.WeightRegularizer(l1=0., l2=0.)
```

```python
keras.regularizers.ActivityRegularizer(l1=0., l2=0.)
```

## 缩写

```keras.regularizers```支持以下缩写

* l1(l=0.01)：L1正则项，又称LASSO

* l2(l=0.01)：L2正则项，又称权重衰减或Ridge

* l1l2(l1=0.01, l2=0.01)： L1-L2混合正则项, 又称ElasticNet

* activity_l1(l=0.01)： L1激活值正则项

* activity_l2(l=0.01)： L2激活值正则项

* activity_l1l2(l1=0.01, l2=0.01)： L1+L2激活值正则项

 【Tips】正则项通常用于对模型的训练施加某种约束，L1正则项即L1范数约束，该约束会使被约束矩阵/向量更稀疏。L2正则项即L2范数约束，该约束会使被约束的矩阵/向量更平滑，因为它对脉冲型的值有很大的惩罚。【@Bigmoyan】