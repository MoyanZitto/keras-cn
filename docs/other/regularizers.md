# 正则项

正则项在优化过程中层的参数或层的激活值添加惩罚项，这些惩罚项将与损失函数一起作为网络的最终优化目标

惩罚项基于层进行惩罚，目前惩罚项的接口与层有关，但```Dense, Conv1D, Conv2D, Conv3D```具有共同的接口。

这些层有三个关键字参数以施加正则项：

* ```kernel_regularizer```：施加在权重上的正则项，为```keras.regularizer.Regularizer```对象

* ```bias_regularizer```：施加在偏置向量上的正则项，为```keras.regularizer.Regularizer```对象

* ```activity_regularizer```：施加在输出上的正则项，为```keras.regularizer.Regularizer```对象

## 例子
```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 可用正则项

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(0.)
```

## 开发新的正则项

任何以权重矩阵作为输入并返回单个数值的函数均可以作为正则项，示例：

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg)
```

可参考源代码[keras/regularizer.py](https://github.com/fchollet/keras/blob/master/keras/regularizers.py)
