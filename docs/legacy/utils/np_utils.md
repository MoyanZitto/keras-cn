# numpy工具



## to_categorical
```python
to_categorical(y, nb_classes=None)
```

将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以`categorical_crossentropy`为目标函数的模型中.

###参数

* y: 类别向量
* nb_classes:总共类别数


## convert_kernel
```python
convert_kernel(kernel, dim_ordering='default')
```

将卷积核矩阵(numpy数组)从Theano形式转换为Tensorflow形式,或转换回来(该转化时自可逆的)
