# 性能评估

## 使用方法

性能评估模块提供了一系列用于模型性能评估的函数,这些函数在模型编译时由`metrics`关键字设置

性能评估函数类似与[目标函数](objectives.md), 只不过该性能的评估结果讲不会用于训练.

可以通过字符串来使用域定义的性能评估函数
```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```
也可以自定义一个Theano/TensorFlow函数并使用之
```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```
### 参数

* y_true:真实标签,theano/tensorflow张量
* y_pred:预测值, 与y_true形式相同的theano/tensorflow张量

### 返回值

单个用以代表输出各个数据点上均值的值

## 可用预定义张量

除fbeta_score额外拥有默认参数beta=1外,其他各个性能指标的参数均为y_true和y_pred

* binary_accuracy: 对二分类问题,计算在所有预测值上的平均正确率
* categorical_accuracy:对多分类问题,计算再所有预测值上的平均正确率
* sparse_categorical_accuracy:与`categorical_accuracy`相同,在对稀疏的目标值预测时有用
* top_k_categorical_accracy: 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
* sparse_top_k_categorical_accuracy：与top_k_categorical_accracy作用相同，但适用于稀疏情况

## 定制评估函数

定制的评估函数可以在模型编译时传入,该函数应该以`(y_true, y_pred)`为参数,并返回单个张量,或从`metric_name`映射到`metric_value`的字典,下面是一个示例:

```python
(y_true, y_pred) as arguments and return a single tensor value.

import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

```
