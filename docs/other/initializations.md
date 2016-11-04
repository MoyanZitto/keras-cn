# 初始化方法

初始化方法使用服从某种概率分布的随机权重来初始化Keras的层

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

【Tips】稍后（一两周吧……）我们希望将各个初始化方法的特点总结一下，请继续关注