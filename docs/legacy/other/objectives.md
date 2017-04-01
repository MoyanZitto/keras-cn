# 目标函数objectives

目标函数，或称损失函数，是编译一个模型必须的两个参数之一：

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

可以通过传递预定义目标函数名字指定目标函数，也可以传递一个Theano/TensroFlow的符号函数作为目标函数，该函数对每个数据点应该只返回一个标量值，并以下列两个参数为参数：

* y_true：真实的数据标签，Theano/TensorFlow张量

* y_pred：预测值，与y_true相同shape的Theano/TensorFlow张量

真实的优化目标函数是在各个数据点得到的损失函数值之和的均值

请参考[<font color='#FF0000'>目标实现代码</font>](https://github.com/fchollet/keras/blob/master/keras/objectives.py)获取更多信息

## 可用的目标函数

* mean_squared_error或mse

* mean_absolute_error或mae

* mean_absolute_percentage_error或mape

* mean_squared_logarithmic_error或msle

* squared_hinge

* hinge

* binary_crossentropy（亦称作对数损失，logloss）

* categorical_crossentropy：亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如```(nb_samples, nb_classes)```的二值序列

* sparse_categorical_crossentrop：如上，但接受稀疏标签。注意，使用该函数时仍然需要你的标签与输出值的维度相同，你可能需要在标签数据上增加一个维度：```np.expand_dims(y,-1)```

* kullback_leibler_divergence:从预测值概率分布Q到真值概率分布P的信息增益,用以度量两个分布的差异.

* poisson：即```(predictions - targets * log(predictions))```的均值

* cosine_proximity：即预测值与真实标签的余弦距离平均值的相反数


**注意**: 当使用"categorical_crossentropy"作为目标函数时,标签应该为多类模式,即one-hot编码的向量,而不是单个数值. 可以使用工具中的`to_categorical`函数完成该转换.示例如下:
```python
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, nb_classes=None)
```



【Tips】过一段时间（等我或者谁有时间吧……）我们将把各种目标函数的表达式和常用场景总结一下。
