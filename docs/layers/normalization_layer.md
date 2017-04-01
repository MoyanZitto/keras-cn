# （批）规范化BatchNormalization

## BatchNormalization层
```python
keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```
该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1

### 参数


* axis: 整数，指定要规范化的轴，通常为特征轴。例如在进行```data_format="channels_first```的2D卷积后，一般会设axis=1。
* momentum: 动态均值的动量
* epsilon：大于0的小浮点数，用于防止除0错误
* center: 若设为True，将会将beta作为偏置加上去，否则忽略参数beta
* scale: 若设为True，则会乘以gamma，否则不使用gamma。当下一层是线性的时，可以设False，因为scaling的操作将被下一层执行。
* beta_initializer：beta权重的初始方法
* gamma_initializer: gamma的初始化方法
* moving_mean_initializer: 动态均值的初始化方法
* moving_variance_initializer: 动态方差的初始化方法
* beta_regularizer: 可选的beta正则
* gamma_regularizer: 可选的gamma正则
* beta_constraint: 可选的beta约束
* gamma_constraint: 可选的gamma约束



### 输入shape

任意，当使用本层为模型首层时，指定```input_shape```参数时有意义。

### 输出shape

与输入shape相同

### 参考文献

* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/pdf/1502.03167v3.pdf)

【Tips】BN层的作用

（1）加速收敛
（2）控制过拟合，可以少用或不用Dropout和正则
（3）降低网络对初始化权重不敏感
（4）允许使用较大的学习率
