# 初始化方法

初始化方法定义了对Keras层设置初始化权重的方法

不同的层可能使用不同的关键字来传递初始化方法，一般来说指定初始化方法的关键字是```init```，例如：
```python
model.add(Dense(64, init='uniform'))
```

## 预定义初始化方法

* uniform

* lecun_uniform: 即有输入节点数之平方根放缩后的均匀分布初始化（[<font color='#FF0000'>LeCun 98</font>](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)）.

* normal

* identity：仅用于权值矩阵为方阵的2D层（```shape[0]=shape[1]```）

* orthogonal：仅用于权值矩阵为方阵的2D层（```shape[0]=shape[1]```），参考[<font color='#FF0000'>Saxe et al.</font>](http://arxiv.org/abs/1312.6120)

* zero

* glorot_normal：由扇入扇出放缩后的高斯初始化（[<font color='#FF0000'>Glorot 2010</font>](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf)）

* glorot_uniform

* he_normal：由扇入放缩后的高斯初始化（[<font color='#FF0000'>He et al.,2014</font>](http://arxiv.org/abs/1502.01852)）

* he_uniform

指定初始化方法传入的可以是一个字符串(必须与上面某种预定义方法匹配),也可以是一个可调用的对象.如果传入可调用的对象,则该对象必须包含两个参数:```shape```(待初始化的变量的shape)和```name```(该变量的名字),该可调用对象必须返回一个(Keras)变量,例如```K.variable()```返回的就是这种变量,下面是例子:
```python
from keras import backend as K
import numpy as np

def my_init(shape, name=None):
    value = np.random.random(shape)
    return K.variable(value, name=name)

model.add(Dense(64, init=my_init))
```
你也可以按这种方法使用```keras.initializations```中的函数:
```python
from keras import initializations

def my_init(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

model.add(Dense(64, init=my_init))
```

【Tips】稍后（一两周吧……）我们希望将各个初始化方法的特点总结一下，请继续关注
