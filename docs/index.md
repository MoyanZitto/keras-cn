# Keras:基于Theano和TensorFlow的深度学习库



## 这就是Keras
Keras是一个极简和高度模块化的神经网络库，Keras由纯Python编写而成并基于Theano或Tensorflow。Keras
为支持快速实验而生，如果你有如下需求，请选择Keras：

* 简易和快速的原型设计（keras具有高度模块化，极简，和可扩充特性）
* 支持CNN和RNN，或二者的结合
* 支持任意的链接方案（包括多输入和多输出训练）
* 无缝CPU和GPU切换

Keras适用的Python版本是：Python 2.7-3.5

***

## 设计原则

* 模块性：模型可理解为一个独立的序列或图，完全可配置的模块以最少的代价自由组合在一起。具体而言，网络层、损失函数、优化器、初始化策略、激活函数、正则化方法都是独立的模块，你可以使用它们来构建自己的模型。

* 极简主义：每个模块都应该尽量的简洁。每一段代码都应该在初次阅读时都显得直观易懂。没有黑魔法，因为它将给迭代和创新带来麻烦。

* 易扩展性：添加新模块超级简单的容易，只需要仿照现有的模块编写新的类或函数即可。创建新模块的便利性使得Keras更适合于先进的研究工作。

* 与Python协作：Keras没有单独的模型配置文件类型（作为对比，caffe有），模型由python代码描述，使其更紧凑和更易debug，并提供了扩展的便利性。

***

##快速开始：30s上手Keras

Keras的核心数据结构是“模型”，模型是一种组织网络层的方式。Keras中主要的模型是Sequential模型，Sequential是一系列网络层按顺序构成的栈。你也可以查看[<font color=#FF0000>Keras function API</font>](getting_started/functional_API.md)来查看更复杂的模型

Sequential模型如下
```python
from keras.models import Sequential

model = Sequential()
```
将一些网络层通过<code>.add\(\)</code>堆叠起来，就构成了一个模型：
```python
from keras.layers.core import Dense, Activation

model.add(Dense(output_dim=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))
```
完成模型的搭建后，我们需要使用<code>.compile\(\)</code>方法来编译模型：
```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```	
编译模型时必须指明损失函数和优化器，如果你需要的话，也可以自己定制损失函数。Keras的一个核心理念就是使得事情在简单的同时，保证用户对他们希望做的事情有足够的控制力度（最绝对的控制来自于源代码的可扩展性）
```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```
完成模型编译后，我们在训练数据上按batch进行一定次数的迭代训练，以拟合网络：
```python
model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
```
当然，我们也可以手动将一批批的数据送入网络中训练，这时候需要使用：
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
搭建一个问答系统、图像分类模型，或神经图灵机、word2vec嵌入器就是这么快。支撑深度学习的基本想法本就是简单的，那凭什么我们不能把它的实现也变得超简单呢？

为了更深入的了解Keras，我们建议你查看一下下面的两个tutorial

* [<font color=#FF0000>快速开始Sequntial模型</font>]()
* [<font color=#FF0000>快速开始泛型模型</font>]()

在Keras代码包的examples文件夹里，我们提供了一些更高级的模型：基于记忆网络的问答系统、基于LSTM的文本的文本生成等。

***

##安装

Keras使用了下面的依赖包：

* numpy，scipy

* pyyaml

* HDF5, h5py（可选，仅在模型的save/load函数中使用）

当使用Theano作为后端时：

* [<font color=FF0000>Theano</font>](http://www.deeplearning.net/software/theano/install.html#install)

当使用TensorFlow为后端时：

* [<font color=FF0000>TensorFlow</font>](https://github.com/tensorflow/tensorflow#download-and-setup)

安装Keras时，请<code>cd</code>到Keras的文件夹中，并运行下面的安装命令：
```python
sudo python setup.py install
```	
你也可以使用PyPI来安装Keras
```python
sudo pip install keras
```
***
	
##在Theano和TensorFlow间切换

Keras默认使用Theano作为后端来进行张量操作，如需切换到TensorFlow，请查看[<font color=FF0000>这里</font>](backend.md)

***

##技术支持

你可以在[<font color=FF0000>Keras Google group</font>](https://groups.google.com/forum/#!forum/keras-users)里提问以获得帮助，如果你生活在中国大陆的话，梯子请自备

你也可以在[<font color=FF0000>Github issues</font>](https://github.com/fchollet/keras/issues)里提问。在提问之前请确保你阅读过我们的[<font color=FF0000>指导</font>](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md)

同时，我们也欢迎同学们加我们的QQ群119427073进行讨论（潜水和灌水会被T哦）


***

## 关于Keras-CN

本文档是Keras文档的中文版，包括[<font color=#FF0000>keras.io</font>](http://keras.io/)的全部内容，以及更多的例子、解释和建议，目前，文档的计划是：

* 现有keras.io文档的中文翻译

* 添加更多的example和更详细的代码说明

* 添加深度学习的一些FAQ，帮助新手更快入坑

欢迎各位指出错漏、不足之处，以便我们改进。欢迎发送邮件到moyan_work@foxmail.com与我取得联系。

文档由下列贡献者贡献：BigMoyan

***

## 为什么叫Keras

Keras (κέρας)是希腊语中号角的意思，它的来源……好吧你真的感兴趣的话请去[<font color=FF0000>keras.io</font>](http://keras.io/)相应页面的最后方查看，我就不翻译这些神话故事了。

