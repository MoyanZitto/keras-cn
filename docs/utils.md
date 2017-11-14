# utils 工具

本模块提供了一系列有用工具


##CustomObjectScope
```python
keras.utils.generic_utils.CustomObjectScope()
```
提供定制类的作用域，在该作用域内全局定制类能够被更改，但在作用域结束后将回到初始状态。
以```with```声明开头的代码将能够通过名字访问定制类的实例，在with的作用范围，这些定制类的变动将一直持续，在with作用域结束后，全局定制类的实例将回归其在with作用域前的状态。

```python
with CustomObjectScope({"MyObject":MyObject}):
    layer = Dense(..., W_regularizer="MyObject")
    # save, load, etc. will recognize custom object by name
```
***

## HDF5Matrix

```python
keras.utils.io_utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

这是一个使用HDF5数据集代替Numpy数组的方法

提供```start```和```end```参数可以进行切片，另外，还可以提供一个正规化函数或匿名函数，该函数将会在每片数据检索时自动调用。

```python
x_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(x_data)
```

* datapath: 字符串，HDF5文件的路径
* dataset: 字符串，在datapath路径下HDF5数据库名字
* start: 整数，想要的数据切片起点
* end: 整数，想要的数据切片终点
* normalizer: 在每个切片数据检索时自动调用的函数对象

***

## Sequence
```
keras.utils.data_utils.Sequence()
```
序列数据的基类，例如一个数据集。
每个Sequence必须实现`__getitem__`和`__len__`方法

下面是一个例子：
```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np

__Here, `x_set` is list of path to the images__

# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):
def __init__(self, x_set, y_set, batch_size):
    self.X,self.y = x_set,y_set
    self.batch_size = batch_size

def __len__(self):
    return len(self.X) // self.batch_size

def __getitem__(self,idx):
    batch_x = self.X[idx*self.batch_size:(idx+1)*self.batch_size]
    batch_y = self.y[idx*self.batch_size:(idx+1)*self.batch_size]

    return np.array([
    resize(imread(file_name), (200,200))
       for file_name in batch_x]), np.array(batch_y)

```

***

## to_categorical
```python
to_categorical(y, num_classes=None)
```

将类别向量(从0到nb_classes的整数向量)映射为二值类别矩阵, 用于应用到以`categorical_crossentropy`为目标函数的模型中.

###参数

* y: 类别向量
* num_classes:总共类别数

***

## normalize
```python
normalize(x, axis=-1, order=2)
```

对numpy数组规范化，返回规范化后的数组

###参数
* x：待规范化的数据
* axis: 规范化的轴
* order：规范化方法，如2为L2范数

***

### custom_object_scope
```python
custom_object_scope()
```
提供定制类的作用域，在该作用域内全局定制类能够被更改，但在作用域结束后将回到初始状态。
以```with```声明开头的代码将能够通过名字访问定制类的实例，在with的作用范围，这些定制类的变动将一直持续，在with作用域结束后，全局定制类的实例将回归其在with作用域前的状态。

本函数返回```CustomObjectScope```对象

```python
with custom_object_scope({"MyObject":MyObject}):
	layer = Dense(..., W_regularizer="MyObject")
	# save, load, etc. will recognize custom object by name
```

***

### get_custom_objects
```python
get_custom_objects()
```

检索全局定制类，推荐利用custom_object_scope更新和清理定制对象，但```get_custom_objects```可被直接用于访问```_GLOBAL_CUSTOM_OBJECTS```。本函数返回从名称到类别映射的全局字典。

```python
get_custom_objects().clear()
get_custom_objects()["MyObject"] = MyObject
```
***

## convert_all_kernels_in_model
```python
convert_all_kernels_in_model(model)
```

将模型中全部卷积核在Theano和TensorFlow模式中切换

***

### plot_model
```python
plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True)
```
绘制模型的结构图

***

### serialize_keras_object
```python
serialize_keras_object(instance)
```
将keras对象序列化

***

### deserialize_keras_object
```python
eserialize_keras_object(identifier, module_objects=None, custom_objects=None, printable_module_name='object')
```
从序列中恢复keras对象

***

### get_file

```python
get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
```

从给定的URL中下载文件, 可以传递MD5值用于数据校验(下载后或已经缓存的数据均可)

默认情况下文件会被下载到`~/.keras`中的`cache_subdir`文件夹，并将其文件名设为`fname`，因此例如一个文件`example.txt`最终将会被存放在`~/.keras/datasets/example.txt~

tar,tar.gz.tar.bz和zip格式的文件可以被提取，提供哈希码可以在下载后校验文件。命令喊程序`shasum`和`sha256sum`可以计算哈希值。


### 参数

* fname: 文件名，如果指定了绝对路径`/path/to/file.txt`,则文件将会保存到该位置。

* origin: 文件的URL地址

* untar: 布尔值,是否要进行解压

* md5_hash: MD5哈希值,用于数据校验，支持`sha256`和`md5`哈希

* cache_subdir: 用于缓存数据的文件夹，若指定绝对路径`/path/to/folder`则将存放在该路径下。

* hash_algorithm: 选择文件校验的哈希算法，可选项有'md5', 'sha256', 和'auto'. 默认'auto'自动检测使用的哈希算法
* extract: 若为True则试图提取文件，例如tar或zip tries extracting the file as an Archive, like tar or zip.
* archive_format: 试图提取的文件格式，可选为'auto', 'tar', 'zip', 和None. 'tar' 包括tar, tar.gz, tar.bz文件. 默认'auto'是['tar', 'zip']. None或空列表将返回没有匹配。 
* cache_dir: 缓存文件存放地在，参考[FAQ](for_beginners/FAQ/#where_config)
### 返回值

下载后的文件地址


### multi_gpu_model
```python
keras.utils.multi_gpu_model(model, gpus)
```

将模型在多个GPU上复制

特别地，该函数用于单机多卡的数据并行支持，它按照下面的方式工作：

（1）将模型的输入分为多个子batch
（2）在每个设备上调用各自的模型，对各自的数据集运行
（3）将结果连接为一个大的batch（在CPU上）

例如，你的batch_size是64而gpus=2，则输入会被分为两个大小为32的子batch，在两个GPU上分别运行，通过连接后返回大小为64的结果。 该函数线性的增加了训练速度，最高支持8卡并行。

该函数只能在tf后端下使用

参数如下：

* model: Keras模型对象，为了避免OOM错误（内存不足），该模型应在CPU上构建，参考下面的例子。
* gpus: 大或等于2的整数，要并行的GPU数目。

该函数返回Keras模型对象，它看起来跟普通的keras模型一样，但实际上分布在多个GPU上。

例子：
```python
import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

# Instantiate the base model
# (here, we do it on CPU, which is optional).
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

# Replicates the model on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# Generate dummy data.
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```
