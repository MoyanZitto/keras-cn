# 性能评估

## 使用方法

性能评估模块提供了一系列用于模型性能评估的函数,这些函数在模型编译时由`metrics`关键字设置

性能评估函数类似与[目标函数](objectives.md), 只不过该性能的评估结果讲不会用于训练.

可以通过字符串来使用域定义的性能评估函数,也可以自定义一个Theano/TensorFlow函数并使用之

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
* mean_squared_error:计算预测值与真值的均方差
* mean_absolute_error:计算预测值与真值的平均绝对误差
* mean_absolute_percentage_error:计算预测值与真值的平均绝对误差率
* mean_squared_logarithmic_error:计算预测值与真值的平均指数误差
* hinge:计算预测值与真值的hinge loss
* squared_hinge:计算预测值与真值的平方hinge loss
* categorical_crossentropy:计算预测值与真值的多类交叉熵(输入值为二值矩阵,而不是向量)
* sparse_categorical_crossentropy:与多类交叉熵相同,适用于稀疏情况
* binary_crossentropy:计算预测值与真值的交叉熵
* poisson:计算预测值与真值的泊松函数值
* cosine_proximity:计算预测值与真值的余弦相似性
* matthews_correlation:计算预测值与真值的马氏距离
* precision：计算精确度，注意percision跟accuracy是不同的。percision用于评价多标签分类中有多少个选中的项是正确的
* recall：召回率，计算多标签分类中有多少正确的项被选中
* fbeta_score:计算F值,即召回率与准确率的加权调和平均,该函数在多标签分类(一个样本有多个标签)时有用,如果只使用准确率作为度量,模型只要把所有输入分类为"所有类别"就可以获得完美的准确率,为了避免这种情况,度量指标应该对错误的选择进行惩罚. F-beta分值(0到1之间)通过准确率和召回率的加权调和平均来更好的度量.当beta为1时,该指标等价于F-measure,beta<1时,模型选对正确的标签更加重要,而beta>1时,模型对选错标签有更大的惩罚.
* fmeasure：计算f-measure，即percision和recall的调和平均

## 定制评估函数

定制的评估函数可以在模型编译时传入,该函数应该以`(y_true, y_pred)`为参数,并返回单个张量,或从`metric_name`映射到`metric_value`的字典,下面是一个示例:

```python
# for custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def false_rates(y_true, y_pred):
    false_neg = ...
    false_pos = ...
    return {
        'false_neg': false_neg,
        'false_pos': false_pos,
    }

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred, false_rates])
```
