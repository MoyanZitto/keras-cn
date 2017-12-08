# Keras:基于Python的深度学习库

## 这就是Keras

Keras是一个高层神经网络API，Keras由纯Python编写而成并基[Tensorflow](https://github.com/tensorflow/tensorflow)、[Theano](https://github.com/Theano/Theano)以及[CNTK](https://github.com/Microsoft/cntk)后端。Keras
为支持快速实验而生，能够把你的idea迅速转换为结果，如果你有如下需求，请选择Keras：

* 简易和快速的原型设计（keras具有高度模块化，极简，和可扩充特性）
* 支持CNN和RNN，或二者的结合
* 无缝CPU和GPU切换

Keras适用的Python版本是：Python 2.7-3.6

Keras的设计原则是

* 用户友好：Keras是为人类而不是天顶星人设计的API。用户的使用体验始终是我们考虑的首要和中心内容。Keras遵循减少认知困难的最佳实践：Keras提供一致而简洁的API， 能够极大减少一般应用下用户的工作量，同时，Keras提供清晰和具有实践意义的bug反馈。
* 模块性：模型可理解为一个层的序列或数据的运算图，完全可配置的模块可以用最少的代价自由组合在一起。具体而言，网络层、损失函数、优化器、初始化策略、激活函数、正则化方法都是独立的模块，你可以使用它们来构建自己的模型。
* 易扩展性：添加新模块超级容易，只需要仿照现有的模块编写新的类或函数即可。创建新模块的便利性使得Keras更适合于先进的研究工作。
* 与Python协作：Keras没有单独的模型配置文件类型（作为对比，caffe有），模型由python代码描述，使其更紧凑和更易debug，并提供了扩展的便利性。

***

## 关于Keras-cn

本文档是Keras文档的中文版，包括[keras.io](http://keras.io/)的全部内容，以及更多的例子、解释和建议

现在，keras-cn的版本号将简单的跟随最新的keras release版本

由于作者水平和研究方向所限，无法对所有模块都非常精通，因此文档中不可避免的会出现各种错误、疏漏和不足之处。如果您在使用过程中有任何意见、建议和疑问，欢迎发送邮件到moyan_work@foxmail.com与我取得联系。

您对文档的任何贡献，包括文档的翻译、查缺补漏、概念解释、发现和修改问题、贡献示例程序等，均会被记录在[致谢](acknowledgement)，十分感谢您对Keras中文文档的贡献！

如果你发现本文档缺失了官方文档的部分内容，请积极联系我补充。

本文档相对于原文档有更多的使用指导和概念澄清，请在使用时关注文档中的Tips，特别的，本文档的额外模块还有：

* Keras新手指南：我们新提供了“Keras新手指南”的页面，在这里我们对Keras进行了感性介绍，并简单介绍了Keras配置方法、一些小知识与使用陷阱，新手在使用前应该先阅读本部分的文档。
* Keras资源：在这个页面，我们罗列一些Keras可用的资源，本页面会不定期更新，请注意关注
* 深度学习与Keras：位于导航栏最下方的该模块翻译了来自Keras作者博客[keras.io](http://blog.keras.io/)
和其他Keras相关博客的文章，该栏目的文章提供了对深度学习的理解和大量使用Keras的例子，您也可以向这个栏目投稿。
所有的文章均在醒目位置标志标明来源与作者，本文档对该栏目文章的原文不具有任何处置权。如您仍觉不妥，请联系本人（moyan_work@foxmail.com）删除。

***

## 当前版本与更新

如果你发现本文档提供的信息有误，有两种可能：

* 你的Keras版本过低：记住Keras是一个发展迅速的深度学习框架，请保持你的Keras与官方最新的release版本相符
* 我们的中文文档没有及时更新：如果是这种情况，请发邮件给我，我会尽快更新

目前文档的版本号是2.0.9，对应于官方的2.0.9 release 版本, 本次更新的主要内容是：

* recurrent新增ConvLSTM2D,SimpleRNNCell, LSTMCell, GRUCell, StackedRNNCells, CuDNNGRE, CuDNNLSTM层
* application中新增了模型InceptionResNetV2
* datasets新增fasion mnist
* FAQ新增Keras的多GPU卡运行指南
* utils新增多卡支持函数multi_gpu_model
* model.compile和model.fit API更新
* 由于年久失修，**深度学习与Keras**栏目中的很多内容的代码已经不再可用，我们决定在新的文档中移除这部分。仍然想访问这些内容（以及已经被移除的一些层，如Maxout）的文档的同学，请下载[中文文档](https://github.com/MoyanZitto/keras-cn)的legacy文件夹，并使用文本编辑器（如sublime）打开对应.md文件。
* 修正了一些错误，感谢@孙永海，@Feng Ying的指正
* 此外，感谢@zh777k制作了Keras2.0.4中文文档的离线版本，对于许多用户而言，这个版本的keras对大多数用户而言已经足够使用了。下载地址在[百度云盘](http://pan.baidu.com/s/1geHmOpH)

注意，keras在github上的master往往要高于当前的release版本，如果你从源码编译keras，可能某些模块与文档说明不相符，请以官方Github代码为准

***

## 快速开始：30s上手Keras

Keras的核心数据结构是“模型”，模型是一种组织网络层的方式。Keras中主要的模型是Sequential模型，Sequential是一系列网络层按顺序构成的栈。你也可以查看[函数式模型](getting_started/functional_API.md)来学习建立更复杂的模型

Sequential模型如下

```python
from keras.models import Sequential

model = Sequential()
```

将一些网络层通过<code>.add\(\)</code>堆叠起来，就构成了一个模型：

```python
from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))
```

完成模型的搭建后，我们需要使用<code>.compile\(\)</code>方法来编译模型：

```python
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
```

编译模型时必须指明损失函数和优化器，如果你需要的话，也可以自己定制损失函数。Keras的一个核心理念就是简明易用，同时保证用户对Keras的绝对控制力度，用户可以根据自己的需要定制自己的模型、网络层，甚至修改源代码。

```python
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))
```

完成模型编译后，我们在训练数据上按batch进行一定次数的迭代来训练网络

```python
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

当然，我们也可以手动将一个个batch的数据送入网络中训练，这时候需要使用：

```python
model.train_on_batch(x_batch, y_batch)
```

随后，我们可以使用一行代码对我们的模型进行评估，看看模型的指标是否满足我们的要求：

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

或者，我们可以使用我们的模型，对新的数据进行预测：

```python
classes = model.predict(x_test, batch_size=128)
```

搭建一个问答系统、图像分类模型，或神经图灵机、word2vec词嵌入器就是这么快。支撑深度学习的基本想法本就是简单的，现在让我们把它的实现也变的简单起来！

为了更深入的了解Keras，我们建议你查看一下下面的两个tutorial

* [快速开始Sequntial模型](getting_started/sequential_model)
* [快速开始函数式模型](getting_started/functional_API)

还有我们的新手教程，虽然是面向新手的，但我们阅读它们总是有益的：

* [Keras新手指南](for_beginners/concepts)

在Keras代码包的examples文件夹里，我们提供了一些更高级的模型：基于记忆网络的问答系统、基于LSTM的文本的文本生成等。

***

## 安装

Keras使用了下面的依赖包，三种后端必须至少选择一种，我们建议选择tensorflow。

* numpy，scipy
* pyyaml
* HDF5, h5py（可选，仅在模型的save/load函数中使用）
* 如果使用CNN的推荐安装cuDNN

当使用TensorFlow为后端时：

* [TensorFlow](https://www.tensorflow.org/install/)

当使用Theano作为后端时：

* [Theano](http://deeplearning.net/software/theano/install.html#install)

当使用CNTK作为后端时：

* [CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine)


“后端”翻译自backend，指的是Keras依赖于完成底层的张量运算的软件包。

从源码安装Keras时，首先git clone keras的代码：

```sh
git clone https://github.com/fchollet/keras.git
```

接着 `cd` 到Keras的文件夹中，并运行下面的安装命令：

```python
sudo python setup.py install
```

你也可以使用PyPI来安装Keras

```python
sudo pip install keras
```

如果你用的是virtualenv虚拟环境，不要用sudo就好。

**详细的Windows和Linux安装教程请参考“Keras新手指南”中给出的安装教程，特别鸣谢SCP-173编写了这些教程**

***
	
## 在Theano、CNTK、TensorFlow间切换

Keras默认使用TensorFlow作为后端来进行张量操作，如需切换到Theano，请查看[这里](backend)

***

## 技术支持

你可以在下列网址提问或加入Keras开发讨论:

- [Keras Google group](https://groups.google.com/forum/#!forum/keras-users)
- [Keras Slack channel](https://kerasteam.slack.com/),[点击这里](https://keras-slack-autojoin.herokuapp.com/)获得邀请.

你也可以在[Github issues](https://github.com/fchollet/keras/issues)里提问或请求新特性。在提问之前请确保你阅读过我们的[指导](https://github.com/fchollet/keras/blob/master/CONTRIBUTING.md)

另外，对于习惯中文的用户，我们推荐在[“集智”平台](https://jizhi.im/index)提问，该平台由Kaiser等搭建，支持在线代码运行环境，我本人会经常访问该网站解答问题

最后，我们也欢迎同学们加我们的QQ群119427073进行讨论（潜水和灌水会被T，入群说明公司/学校-职位/年级）

***

## 小额赞助

如果你觉得本文档对你的研究和使用有所帮助，欢迎扫下面的二维码对作者进行小额赞助，以鼓励作者进一步完善文档内容，提高文档质量。同时，不妨为[本文档的github](https://github.com/MoyanZitto/keras-cn)加颗星哦

![付款二维码](images/moyan.png)
