# 约束项

来自```constraints```模块的函数在优化过程中为网络的参数施加约束

惩罚项基于层进行惩罚，目前惩罚项的接口与层有关，但```Dense, Conv1D, Conv2D, Conv3D```具有共同的接口。

这些层通过一下关键字施加约束项

* ```kernel_constraint```：对主权重矩阵进行约束

* ```bias_constraint```：对偏置向量进行约束

```python
from keras.constraints import maxnorm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## 预定义约束项

* max_norm(m=2)：最大模约束

* non_neg()：非负性约束

* unit_norm()：单位范数约束, 强制矩阵沿最后一个轴拥有单位范数

* min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0): 最小/最大范数约束