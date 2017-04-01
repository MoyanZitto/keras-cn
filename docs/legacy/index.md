# Keras:基于Theano和TensorFlow的深度学习库



## 这就是Keras
Keras是一个高层神经网络库，Keras由纯Python编写而成并基Tensorflow或Theano。Keras
为支持快速实验而生，能够把你的idea迅速转换为结果，如果你有如下需求，请选择Keras：

* 简易和快速的原型设计（keras具有高度模块化，极简，和可扩充特性）
* 支持CNN和RNN，或二者的结合
* 支持任意的链接方案（包括多输入和多输出训练）
* 无缝CPU和GPU切换

Keras适用的Python版本是：Python 2.7-3.5

Keras的设计原则是

* 模块性：模型可理解为一个独立的序列或图，完全可配置的模块以最少的代价自由组合在一起。具体而言，网络层、损失函数、优化器、初始化策略、激活函数、正则化方法都是独立的模块，你可以使用它们来构建自己的模型。

* 极简主义：每个模块都应该尽量的简洁。每一段代码都应该在初次阅读时都显得直观易懂。没有黑魔法，因为它将给迭代和创新带来麻烦。

* 易扩展性：添加新模块超级简单的容易，只需要仿照现有的模块编写新的类或函数即可。创建新模块的便利性使得Keras更适合于先进的研究工作。

* 与Python协作：Keras没有单独的模型配置文件类型（作为对比，caffe有），模型由python代码描述，使其更紧凑和更易debug，并提供了扩展的便利性。

Keras从2015年3月开始启动，经过一年多的开发，目前Keras进入了1.0的时代。Keras 1.0依然遵循相同的设计原则，但与之前的版本相比有很大的不同。如果你曾经使用过此前的其他版本Keras。你或许会关心1.0的新特性。

* 泛型模型：简单和强大的新模块，用于支持复杂深度学习模型的搭建。

* 更优秀的性能：现在，Keras模型的编译时间得到缩短。所有的RNN现在都可以用两种方式实现，以供用户在不同配置任务和配置环境下取得最大性能。现在，基于Theano的RNN也可以被展开，以获得大概25%的加速计算。

* 测量指标：现在，你可以提供一系列的测量指标来在Keras的任何监测点观察模型性能。

* 更优的用户体验：我们面向使用者重新编写了代码，使得函数API更简单易记，同时提供更有效的出错信息。

* 新版本的Keras提供了Lambda层，以实现一些简单的计算任务。

* ...

如果你已经基于Keras0.3编写了自己的层，那么在升级后，你需要为自己的代码做以下调整，以在Keras1.0上继续运行。请参考[<font color='#FF0000'>编写自己的层</font>](layers/writting_layer/#adjust)

***

## 关于Keras-cn

本文档是Keras文档的中文版，包括[<font color=#FF0000>keras.io</font>](http://keras.io/)的全部内容，以及更多的例子、解释和建议

现在，keras-cn的版本号将简单的跟随最新的keras release版本

由于作者水平和研究方向所限，无法对所有模块都非常精通，因此文档中不可避免的会出现各种错误、疏漏和不足之处。如果您在使用过程中有任何意见、建议和疑问，欢迎发送邮件到moyan_work@foxmail.com与我取得联系。

您对文档的任何贡献，包括文档的翻译、查缺补漏、概念解释、发现和修改问题、贡献示例程序等，均会被记录在[<font color='FF0000'>致谢</font>](acknowledgement)，十分感谢您对Keras中文文档的贡献！


同时，也欢迎您撰文向本文档投稿，您的稿件被录用后将以单独的页面显示在网站中，您有权在您的网页下设置赞助二维码，以获取来自网友的小额赞助。

如果你发现本文档缺失了官方文档的部分内容，请积极联系我补充。


本文档相对于原文档有更多的使用指导和概念澄清，请在使用时关注文档中的Tips，特别的，本文档的额外模块还有：

* 一些基本概念：位于快速开始模块的[<font color='#FF0000'>一些基本概念</font>](getting_started/concepts)简单介绍了使用Keras前需要知道的一些小知识，新手在使用前应该先阅读本部分的文档。

* Keras安装和配置指南，提供了详细的Linux和Windows下Keras的安装和配置步骤。

* 深度学习与Keras：位于导航栏最下方的该模块翻译了来自Keras作者博客[<font color='#FF0000'>keras.io</font>](http://blog.keras.io/)和其他Keras相关博客的文章，该栏目的文章提供了对深度学习的理解和大量使用Keras的例子，您也可以向这个栏目投稿。
所有的文章均在醒目位置标志标明来源与作者，本文档对该栏目文章的原文不具有任何处置权。如您仍觉不妥，请联系本人（moyan_work@foxmail.com）删除。

***

## 当前版本与更新

如果你发现本文档提供的信息有误，有两种可能：

* 你的Keras版本过低：记住Keras是一个发展迅速的深度学习框架，请保持你的Keras与官方最新的release版本相符

* 我们的中文文档没有及时更新：如果是这种情况，请发邮件给我，我会尽快更新

目前文档的版本号是1.2.1，对应于官方的1.2.1 release 版本, 本次更新的主要内容是：

* 为Application中的图片分类模型增加了`classes`和`input_shape`，但`classes`的说明在原文档中缺失
* 暂时移除了SpatialDropout1D,SpatialDropout2D,SpatialDropout3D的文档，但它们的源代码仍然保留在个keras中，你仍然可以用这些layer
* 1.2.0和当前版本均可使用ConvLSTM2D层，这个层在代码中有说明但没有体现在文档中，我们暂时也不提供这个层的说明，如需使用请查看源代码中的说明
* 增加了AC-GAN的例子

注意，keras在github上的master往往要高于当前的release版本，如果你从源码编译keras，可能某些模块与文档说明不相符，请以官方Github代码为准

***

##快速开始：30s上手Keras

Keras的核心数据结构是“模型”，模型是一种组织网络层的方式。Keras中主要的模型是Sequential模型，Sequential是一系列网络层按顺序构成的栈。你也可以查看[<font color=#FF0000>泛型模型</font>](getting_started/functional_API.md)来学习建立更复杂的模型

Sequential模型如下
```python
from keras.models import Sequential

model = Sequential()
```
将一些网络层通过<code>.add\(\)</code>堆叠起来，就构成了一个模型：
```python
from keras.layers import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
```
完成模型的搭建后，我们需要使用<code>.compile\(\)</code>方法来编译模型：
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```	
编译模型时必须指明损失函数和优化器，如果你需要的话，也可以自己定制损失函数。Keras的一个核心理念就是简明易用同时，保证用户对Keras的绝对控制力度，用户可以根据自己的需要定制自己的模型、网络层，甚至修改源代码。
```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```
完成模型编译后，我们在训练数据上按batch进行一定次数的迭代训练，以拟合网络，关于为什么要使用‘batch’，请参考[<font color=#FF0000>一些基本概念</font>](getting_started/concepts/#batch)


```python
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```
当然，我们也可以手动将一个个batch的数据送入网络中训练，这时候需要使用：
```python
model.train_on_batch(X_batch, Y_batch)
```
随后，我们可以使用一行代码对我们的模型进行评估，看看模型的指标是否满足我们的要求：
```python
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=32)
```	
或者，我们可以使用我们的模型，对新的数据进行预测：
```python
classes = model.predict_classes(X_test, batch_size=32)
proba = model.predict_proba(X_test, batch_size=32)
```	
搭建一个问答系统、图像分类模型，或神经图灵机、word2vec词嵌入器就是这么快。支撑深度学习的基本想法本就是简单的，现在让我们把它的实现也变的简单起来！

为了更深入的了解Keras，我们建议你查看一下下面的两个tutorial

* [<font color=#FF0000>快速开始Sequntial模型</font>](getting_started/sequential_model)
* [<font color=#FF0000>快速开始泛型模型</font>](getting_started/functional_API)

还有我们对一些概念的解释

* [<font color=#FF0000>一些基本概念</font>](getting_started/concepts)

在Keras代码包的examples文件夹里，我们提供了一些更高级的模型：基于记忆网络的问答系统、基于LSTM的文本的文本生成等。

***

##安装

Keras使用了下面的依赖包：

* numpy，scipy

* pyyaml

* HDF5, h5py（可选，仅在模型的save/load函数中使用）

当使用TensorFlow为后端时：

* [<font color=FF0000>TensorFlow</font>](https://github.com/tensorflow/tensorflow#download-and-setup)

当使用Theano作为后端时：

* [<font color=FF0000>Theano</font>](http://www.deeplearning.net/software/theano/install.html#install)


【Tips】“后端”翻译自backend，指的是Keras依赖于完成底层的张量运算的软件包。【@Bigmoyan】

安装Keras时，请<code>cd</code>到Keras的文件夹中，并运行下面的安装命令：
```python
sudo python setup.py install
```	
你也可以使用PyPI来安装Keras
```python
sudo pip install keras
```

**详细的Windows和Linux安装教程请参考“快速开始”一节中给出的安装教程，特别鸣谢SCP-173编写了这些教程**

***
	
##在Theano和TensorFlow间切换

Keras默认使用TensorFlow作为后端来进行张量操作，如需切换到Theano，请查看[<font color=FF0000>这里</font>](backend)

***

##技术支持

你可以在下列网址提问或加入Keras开发讨论:

- [<font color=FF0000>Keras Google group</font>](https://groups.google.com/forum/#!forum/keras-users)
- [<font color=FF0000>Keras Slack channel</font>](https://kerasteam.slack.com/),[<font color=FF0000>点击这里</font>]获得邀请.

你也可以在[<font color=FF0000>Github issues</font>](https://github.com/fchollet/keras/issues)里提问或请求新特性。在提问之前请确保你阅读过我们的[<font color=FF0000>指导</font>](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md)

同时，我们也欢迎同学们加我们的QQ群119427073进行讨论（潜水和灌水会被T，入群说明公司/学校-职位/年级）

***

## 小额赞助

如果你觉得本文档对你的研究和使用有所帮助，欢迎扫下面的二维码对作者进行小额赞助，以鼓励作者进一步完善文档内容，提高文档质量。同时，不妨为[<font color='#FF0000'>本文档的github</font>](https://github.com/MoyanZitto/keras-cn)加颗星哦

![付款二维码](images/moyan.png)

**如果你觉得有用页面下有小额赞助的二维码或微信/支付宝账号，说明该页面由其他作者贡献，要对他们进行小额赞助请使用该页面下的二维码或账号**




