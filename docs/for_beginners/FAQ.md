# Keras FAQ：常见问题

* [如何引用Keras？](#citation)
* [如何使Keras调用GPU？](#GPU)
* [如何在多张GPU卡上使用Keras](#multi-GPU)
* ["batch", "epoch"和"sample"都是啥意思？](#batch)
* [如何保存Keras模型？](#save_model)
* [为什么训练误差(loss)比测试误差高很多？](#loss)
* [如何获取中间层的输出？](#intermediate_layer)
* [如何利用Keras处理超过机器内存的数据集？](#dataset)
* [当验证集的loss不再下降时，如何中断训练？](#stop_train)
* [验证集是如何从训练集中分割出来的？](#validation_spilt)
* [训练数据在训练时会被随机洗乱吗？](#shuffle)
* [如何在每个epoch后记录训练/测试的loss和正确率？](#history)
* [如何使用状态RNN（statful RNN）？](#statful_RNN)
* [如何“冻结”网络的层？](#freeze)
* [如何从Sequential模型中去除一个层？](#pop)
* [如何在Keras中使用预训练的模型](#pretrain)
* [如何在Keras中使用HDF5输入？](#hdf5)
* [Keras的配置文件存储在哪里？](#where_config)
* [在使用Keras开发过程中，我如何获得可复现的结果？](#reproduce)
***

<a name='citation'>
<font color='#404040'>
## 如何引用Keras？
</font>
</a>

如果Keras对你的研究有帮助的话，请在你的文章中引用Keras。这里是一个使用BibTex的例子

```python
@misc{chollet2015keras,
  author = {Chollet, François and others},
  title = {Keras},
  year = {2015},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fchollet/keras}}
}
```

***

<a name='GPU'>
<font color='#404040'>
## 如何使Keras调用GPU？
</font>
</a>

如果采用TensorFlow作为后端，当机器上有可用的GPU时，代码会自动调用GPU进行并行计算。如果使用Theano作为后端，可以通过以下方法设置：

方法1：使用Theano标记

在执行python脚本时使用下面的命令：

```python
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

方法2：设置```.theano```文件

点击[这里](http://deeplearning.net/software/theano/library/config.html)查看指导教程
	
方法3：在代码的开头处手动设置```theano.config.device```和```theano.config.floatX```

```python
	import theano
	theano.config.device = 'gpu'
	theano.config.floatX = 'float32'
```


<a name='multi-GPU'>
<font color='#404040'>
## 如何在多张GPU卡上使用Keras？
</font>
</a>

我们建议有多张GPU卡可用时，使用TnesorFlow后端。

有两种方法可以在多张GPU上运行一个模型：数据并行/设备并行

大多数情况下，你需要的很可能是“数据并行”


### 数据并行

数据并行将目标模型在多个设备上各复制一份，并使用每个设备上的复制品处理整个数据集的不同部分数据。Keras在`keras.utils.multi_gpu_model`中提供有内置函数，该函数可以产生任意模型的数据并行版本，最高支持在8片GPU上并行。
请参考[utils](../utils.md)中的multi_gpu_model文档。 下面是一个例子：

```python
from keras.utils import multi_gpu_model

# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# This `fit` call will be distributed on 8 GPUs.
# Since the batch size is 256, each GPU will process 32 samples.
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

### 设备并行

设备并行是在不同设备上运行同一个模型的不同部分，当模型含有多个并行结构，例如含有两个分支时，这种方式很适合。

这种并行方法可以通过使用TensorFlow device scopes实现，下面是一个例子：


```python
# Model where a shared LSTM is used to encode two different sequences in parallel
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# Process the first sequence on one GPU
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# Process the next sequence on another GPU
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# Concatenate results on CPU
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```
***

<a name='batch'>
<font color='#404040'>
## "batch", "epoch"和"sample"都是啥意思？？
</font>
</a>

下面是一些使用keras时常会遇到的概念，我们来简单解释。

- Sample：样本，数据集中的一条数据。例如图片数据集中的一张图片，语音数据中的一段音频。
- Batch：中文为批，一个batch由若干条数据构成。batch是进行网络优化的基本单位，网络参数的每一轮优化需要使用一个batch。batch中的样本是被并行处理的。与单个样本相比，一个batch的数据能更好的模拟数据集的分布，batch越大则对输入数据分布模拟的越好，反应在网络训练上，则体现为能让网络训练的方向“更加正确”。但另一方面，一个batch也只能让网络的参数更新一次，因此网络参数的迭代会较慢。在测试网络的时候，应该在条件的允许的范围内尽量使用更大的batch，这样计算效率会更高。
- Epoch，epoch可译为“轮次”。如果说每个batch对应网络的一次更新的话，一个epoch对应的就是网络的一轮更新。每一轮更新中网络更新的次数可以随意，但通常会设置为遍历一遍数据集。因此一个epoch的含义是模型完整的看了一遍数据集。
	设置epoch的主要作用是把模型的训练的整个训练过程分为若干个段，这样我们可以更好的观察和调整模型的训练。Keras中，当指定了验证集时，每个epoch执行完后都会运行一次验证集以确定模型的性能。另外，我们可以使用回调函数在每个epoch的训练前后执行一些操作，如调整学习率，打印目前模型的一些信息等，详情请参考Callback一节。
***

<a name='save_model'>
<font color='#404040'>
## 如何保存Keras模型？
</font>
</a>

我们不推荐使用pickle或cPickle来保存Keras模型

你可以使用```model.save(filepath)```将Keras模型和权重保存在一个HDF5文件中，该文件将包含：

* 模型的结构，以便重构该模型
* 模型的权重
* 训练配置（损失函数，优化器等）
* 优化器的状态，以便于从上次训练中断的地方开始

使用```keras.models.load_model(filepath)```来重新实例化你的模型，如果文件中存储了训练配置的话，该函数还会同时完成模型的编译

例子：
```python
from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')
```
如果你只是希望保存模型的结构，而不包含其权重或配置信息，可以使用：
```python
# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()
```
这项操作将把模型序列化为json或yaml文件，这些文件对人而言也是友好的，如果需要的话你甚至可以手动打开这些文件并进行编辑。

当然，你也可以从保存好的json文件或yaml文件中载入模型：

```python
# model reconstruction from JSON:
from keras.models import model_from_json
model = model_from_json(json_string)

# model reconstruction from YAML
model = model_from_yaml(yaml_string)
```

如果需要保存模型的权重，可通过下面的代码利用HDF5进行保存。注意，在使用前需要确保你已安装了HDF5和其Python库h5py

```
model.save_weights('my_model_weights.h5')
```	

如果你需要在代码中初始化一个完全相同的模型，请使用：
```python
model.load_weights('my_model_weights.h5')
```
如果你需要加载权重到不同的网络结构（有些层一样）中，例如fine-tune或transfer-learning，你可以通过层名字来加载模型：

```python
model.load_weights('my_model_weights.h5', by_name=True)
```

例如：
```python
"""
假如原模型为：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name="dense_1"))
    model.add(Dense(3, name="dense_2"))
    ...
    model.save_weights(fname)
"""
# new model
model = Sequential()
model.add(Dense(2, input_dim=3, name="dense_1"))  # will be loaded
model.add(Dense(10, name="new_dense"))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model.load_weights(fname, by_name=True)

```
***

<a name='loss'>
<font color='#404040'>
## 为什么训练误差比测试误差高很多？
</font>
</a>

一个Keras的模型有两个模式：训练模式和测试模式。一些正则机制，如Dropout，L1/L2正则项在测试模式下将不被启用。

另外，训练误差是训练数据每个batch的误差的平均。在训练过程中，每个epoch起始时的batch的误差要大一些，而后面的batch的误差要小一些。另一方面，每个epoch结束时计算的测试误差是由模型在epoch结束时的状态决定的，这时候的网络将产生较小的误差。

【Tips】可以通过定义回调函数将每个epoch的训练误差和测试误差并作图，如果训练误差曲线和测试误差曲线之间有很大的空隙，说明你的模型可能有过拟合的问题。当然，这个问题与Keras无关。

***

<a name='intermediate_layer'>
<font color='#404040'>
## 如何获取中间层的输出？
</font>
</a>

一种简单的方法是创建一个新的`Model`，使得它的输出是你想要的那个输出

```python
from keras.models import Model

model = ...  # create the original model

layer_name = 'my_layer'
intermediate_layer_model = Model(input=model.input,
                                 output=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data
```

此外，我们也可以建立一个Keras的函数来达到这一目的：

```python
from keras import backend as K

# with a Sequential model
get_3rd_layer_output = K.function([model.layers[0].input],
								  [model.layers[3].output])
layer_output = get_3rd_layer_output([X])[0]
```
当然，我们也可以直接编写Theano和TensorFlow的函数来完成这件事

注意，如果你的模型在训练和测试两种模式下不完全一致，例如你的模型中含有Dropout层，批规范化（BatchNormalization）层等组件，你需要在函数中传递一个learning_phase的标记，像这样：
```
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
								  [model.layers[3].output])

# output in test mode = 0
layer_output = get_3rd_layer_output([X, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([X, 1])[0]
```

***

<a name='dataset'>
<font color='#404040'>
## 如何利用Keras处理超过机器内存的数据集？
</font>
</a>

可以使用```model.train_on_batch(X,y)```和```model.test_on_batch(X,y)```。请参考[模型](../models/sequential.md)

另外，也可以编写一个每次产生一个batch样本的生成器函数，并调用```model.fit_generator(data_generator, samples_per_epoch, nb_epoch)```进行训练

这种方式在Keras代码包的example文件夹下CIFAR10例子里有示范，也可点击[这里](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)在github上浏览。

***

<a name='early_stopping'>
<font color='#404040'>
## 当验证集的loss不再下降时，如何中断训练？
</font>
</a>

可以定义```EarlyStopping```来提前终止训练
```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])
```
请参考[回调函数](../other/callbacks)


***

<a name='validation_spilt'>
<font color='#404040'>
## 验证集是如何从训练集中分割出来的？
</font>
</a>

如果在```model.fit```中设置```validation_spilt```的值，则可将数据分为训练集和验证集，例如，设置该值为0.1，则训练集的最后10%数据将作为验证集，设置其他数字同理。注意，原数据在进行验证集分割前并没有被shuffle，所以这里的验证集严格的就是你输入数据最末的x%。


***

<a name='shuffle'>
<font color='#404040'>
## 训练数据在训练时会被随机洗乱吗？
</font>
</a>

是的，如果```model.fit```的```shuffle```参数为真，训练的数据就会被随机洗乱。不设置时默认为真。训练数据会在每个epoch的训练中都重新洗乱一次。

验证集的数据不会被洗乱


***

<a name='history'>
<font color='#404040'>
## 如何在每个epoch后记录训练/测试的loss和正确率？
</font>
</a>

```model.fit```在运行结束后返回一个```History```对象，其中含有的```history```属性包含了训练过程中损失函数的值以及其他度量指标。
```python
hist = model.fit(X, y, validation_split=0.2)
print(hist.history)
```
	
***

<a name='statful_RNN'>
<font color='#404040'>
## 如何使用状态RNN（statful RNN）？
</font>
</a>

一个RNN是状态RNN，意味着训练时每个batch的状态都会被重用于初始化下一个batch的初始状态。

当使用状态RNN时，有如下假设

* 所有的batch都具有相同数目的样本

* 如果```X1```和```X2```是两个相邻的batch，那么对于任何```i```，```X2[i]```都是```X1[i]```的后续序列

要使用状态RNN，我们需要

* 显式的指定每个batch的大小。可以通过模型的首层参数```batch_input_shape```来完成。```batch_input_shape```是一个整数tuple，例如\(32,10,16\)代表一个具有10个时间步，每步向量长为16，每32个样本构成一个batch的输入数据格式。

* 在RNN层中，设置```stateful=True```

要重置网络的状态，使用：

* ```model.reset_states()```来重置网络中所有层的状态

* ```layer.reset_states()```来重置指定层的状态

例子：
```python
X  # this is our input data, of shape (32, 21, 16)
# we will feed it to our model in sequences of length 10

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# we train the network to predict the 11th timestep given the first 10:
model.train_on_batch(X[:, :10, :], np.reshape(X[:, 10, :], (32, 16)))

# the state of the network has changed. We can feed the follow-up sequences:
model.train_on_batch(X[:, 10:20, :], np.reshape(X[:, 20, :], (32, 16)))

# let's reset the states of the LSTM layer:
model.reset_states()

# another way to do it in this case:
model.layers[0].reset_states()
```
注意，```predict```，```fit```，```train_on_batch```
，```predict_classes```等方法都会更新模型中状态层的状态。这使得你可以不但可以进行状态网络的训练，也可以进行状态网络的预测。

***



<a name='freeze'>
<font color='#404040'>
## 如何“冻结”网络的层？
</font>
</a>

“冻结”一个层指的是该层将不参加网络训练，即该层的权重永不会更新。在进行fine-tune时我们经常会需要这项操作。
在使用固定的embedding层处理文本输入时，也需要这个技术。

可以通过向层的构造函数传递```trainable```参数来指定一个层是不是可训练的，如：

```python
frozen_layer = Dense(32,trainable=False)
```

此外，也可以通过将层对象的```trainable```属性设为```True```或```False```来为已经搭建好的模型设置要冻结的层。
在设置完后，需要运行```compile```来使设置生效，例如：

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # this does NOT update the weights of `layer`
trainable_model.fit(data, labels)  # this updates the weights of `layer`
```

***

<a name='pop'>
<font color='#404040'>
## 如何从Sequential模型中去除一个层？
</font>
</a>

可以通过调用```.pop()```来去除模型的最后一个层，反复调用n次即可去除模型后面的n个层

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```


***

<a name='pretrain'>
<font color='#404040'>
## 如何在Keras中使用预训练的模型？
</font>
</a>

我们提供了下面这些图像分类的模型代码及预训练权重：

- VGG16
- VGG19
- ResNet50
- Inception v3

可通过```keras.applications```载入这些模型：

```python
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

model = VGG16(weights='imagenet', include_top=True)
```

这些代码的使用示例请参考```.Application```模型的[文档](../other/application.md)


使用这些预训练模型进行特征抽取或fine-tune的例子可以参考[此博客](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

VGG模型也是很多Keras例子的基础模型，如：

* [<font color='#FF0000'>Style-transfer</font>](https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py)
* [<font color='#FF0000'>Feature visualization</font>](https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py)
* [<font color='#FF0000'>Deep dream</font>](https://github.com/fchollet/keras/blob/master/examples/deep_dream.py)

<a name='hdf5'>
<font color='#404040'>
## 如何在Keras中使用HDF5输入？
</font>
</a>

你可以使用keras.utils中的```HDF5Matrix```类来读取HDF5输入，参考[这里](../utils.md)

可以直接使用HDF5数据库，示例
```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    X_data = f['X_data']
    model.predict(X_data)
```

***
<a name='where_config'>
<font color='#404040'>
## Keras的配置文件存储在哪里？
</font>
</a>

所有的Keras数据默认存储在：
```bash
$HOME/.keras/
```

对windows用户而言，`$HOME`应替换为`%USERPROFILE%`

当Keras无法在上面的位置创建文件夹时（例如由于权限原因），备用的地址是`/tmp/.keras/`

Keras配置文件为JSON格式的文件，保存在`$HOME/.keras/keras.json`。默认的配置文件长这样：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
该文件包含下列字段：


- 默认的图像数据格式`channels_last`或`channels_first`
- 用于防止除零错误的`epsilon`
- 默认的浮点数类型
- 默认的后端

类似的，缓存的数据集文件，即由`get_file()`下载的文件，默认保存在`$HOME/.keras/datasets/`

***
<a name='reproduce'>
<font color='#404040'>
## 在使用Keras开发过程中，我如何获得可复现的结果?
</font>
</a>

在开发模型中，有时取得可复现的结果是很有用的。例如，这可以帮助我们定位模型性能的改变是由模型本身引起的还是由于数据上的变化引起的。下面的代码展示了如何获得可复现的结果，该代码基于Python3的tensorflow后端

```python
import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Rest of code follows ...
```
