# Keras FAQ：常见问题

* [如何引用Keras？](#citation)
* [如何使Keras调用GPU？](#GPU)
* [如何保存Keras模型？](#save_model)
* [为什么训练误差(loss)比测试误差高很多？](#loss)
* [如何观察中间层的输出？](#intermediate_layer)
* [如何利用Keras处理超过机器内存的数据集？](#dataset)
* [当验证集的loss不再下降时，如何中断训练？](#stop_train)
* [验证集是如何从训练集中分割出来的？](#validation_spilt)
* [训练数据在训练时会被随机洗乱吗？](#shuffle)
* [如何在每个epoch后记录训练/测试的loss和正确率？](#history)
* [如何使用状态RNN？](#statful_RNN)

***

<a name='citation'>
<font color='000000'>
## 如何引用Keras？

如果Keras对你的研究有帮助的话，请在你的文章中引用Keras。这里是一个使用BibTex的例子

	@misc{chollet2015keras,
	  author = {Chollet, François},
	  title = {Keras},
	  year = {2015},
	  publisher = {GitHub},
	  journal = {GitHub repository},
	  howpublished = {\url{https://github.com/fchollet/keras}}
	}
</font>
</a>

***

<a name='GPU'>
<font color='000000'>
## 如何使Keras调用GPU？

如果采用TensorFlow作为后端，当机器上有可用的GPU时，代码会自动调用GPU进行并行计算。如果使用Theano作为后端，可以通过以下方法设置：

方法1：使用Theano标记

在执行python脚本时使用下面的命令：

	THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py

方法2：设置<code>.theano</code>文件

	点击[<font color="FF0000">这里</font>](http://deeplearning.net/software/theano/library/config.html)查看指导教程
	
方法3：在代码的开头处手动设置<code>theano.config.device</code>和<code>theano.config.floatX</code>

	import theano
	theano.config.device = 'gpu'
	theano.config.floatX = 'float32'
</font>
</a>
	
***

<a name='save_model'>
<font color='000000'>
## 如何保存Keras模型？

我们不推荐使用pickle或cPickle来保存Keras模型

如果只需要保存模型的结构，而不需要保存其权重，可以使用

	# save as JSON
	json_string = model.to_json()

	# save as YAML
	yaml_string = model.to_yaml()
	
需要时，可以从保存好的json文件或yaml文件中载入模型：

	# model reconstruction from JSON:
	from keras.models import model_from_json
	model = model_from_json(json_string)

	# model reconstruction from YAML
	model = model_from_yaml(yaml_string)
	
如果需要保存模型的权重，可通过下面的代码利用HDF5进行保存。注意，在使用前需要确保你已安装了HDF5和其Python库h5py

	model.save_weights('my_model_weights.h5')
	
在需要时，你可用保存好的权重初始化相应的模型：

	model.load_weights('my_model_weights.h5')
	
通过模型和权重的保存和加载，我们可以轻松重构已经训练好的模型

	json_string = model.to_json()
	open('my_model_architecture.json', 'w').write(json_string)
	model.save_weights('my_model_weights.h5')

	# elsewhere...
	model = model_from_json(open('my_model_architecture.json').read())
	model.load_weights('my_model_weights.h5')

最后，在模型使用前，还是需要再编译一下

	model.compile(optimizer='adagrad', loss='mse')
</font>
</a>

***

<a name='loss'>
<font color='000000'>
## 为什么训练误差比测试误差高很多？

一个Keras的模型有两个模式：训练模式和测试模式。一些正则机制，如Dropout，L1/L2正则项在测试模式下将不被启用。

另外，训练误差是训练数据每个batch的误差的平均。在训练过程中，每个epoch起始时的batch的误差要大一些，而后面的batch的误差要小一些。另一方面，每个epoch结束时计算的测试误差是由模型在epoch结束时的状态决定的，这时候的网络将产生较小的误差。

【Tips】可以通过定义回调函数将每个epoch的训练误差和测试误差并作图，如果训练误差曲线和测试误差曲线之间有很大的空隙，说明你的模型可能有过拟合的问题。当然，这个问题与Keras无关。
</font>
</a>

***

<a name='intermediate_layer'>
<font color='000000'>
## 如何观察中间层的输出？

我们可以建立一个Keras的函数来将获得给定输入时特定层的输出：

	from keras import backend as K

	# with a Sequential model
	get_3rd_layer_output = K.function([model.layers[0].input],
									  [model.layers[3].output])
	layer_output = get_3rd_layer_output([X])[0]

当然，我们也可以直接编写Theano和TensorFlow的函数来完成这件事

注意，如果你的模型在训练和测试两种模式下不完全一致，例如你的模型中含有Dropout层，批规范化（BatchNormalization）层等组件，你需要在函数中传递一个learning_phase的标记，像这样：

	get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
									  [model.layers[3].output])

	# output in train mode = 0
	layer_output = get_3rd_layer_output([X, 0])[0]

	# output in test mode = 1
	layer_output = get_3rd_layer_output([X, 1])[0]

<font color='#000000'>另一种更灵活的获取中间层输出的方法是使用[<font color="FF0000">泛型模型</font>](functional_API.md)</font>

</font>
</a>

***
<a name='dataset'>
<font color='000000'>
## 如何利用Keras处理超过机器内存的数据集？
<font color='000000'>
可以使用<code>model.train_on_batch(X,y)</code>和<code>model.test_on_batch(X,y)</code>。请参考[<font color='#FF0000'>模型</font>](../models/sequential.md)
</font>
另外，也可以编写一个每次产生一个batch样本的生成器函数，并调用<code>model.fit_generator(data_generator, samples_per_epoch, nb_epoch)</code>进行训练

这种方式在Keras代码包的example文件夹下CIFAR10例子里有示范，也可点击[<font color='#FF0000'>这里</font>](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py)在github上浏览。

</font>
</a>

***

<a name='stop_train'>
<font color='000000'>

## 当验证集的loss不再下降时，如何中断训练？

可以定义<code>EarlyStopping</code>来提前终止训练

	from keras.callbacks import EarlyStopping
	early_stopping = EarlyStopping(monitor='val_loss', patience=2)
	model.fit(X, y, validation_split=0.2, callbacks=[early_stopping])

<font color='000000'>请参考[<font color='#FF0000'>回调函数</font>](../other/callbacks.md)</font>
</font>
</a>

***

<a name='validation_spilt'>
<font color='000000'>

## 验证集是如何从训练集中分割出来的？

如果在<code>model.fit</code>中设置<code>validation_spilt</code>的值，则可将数据分为训练集和验证集，例如，设置该值为0.1，则训练集的最后10%数据将作为验证集，设置其他数字同理。
</font>
</a>

***

<a name='shuffle'>
<font color='000000'>

## 训练数据在训练时会被随机洗乱吗？

是的，如果<code>model.fit</code>的<code>shuffle</code>参数为真，训练的数据就会被随机洗乱。不设置时默认为真。训练数据会在每个epoch的训练中都重新洗乱一次。

验证集的数据不会被洗乱
</font>
</a>

***

<a name='history'>
<font color='000000'>
## 如何在每个epoch后记录训练/测试的loss和正确率？

<code>model.fit</code>在运行结束后返回一个<code>History</code>对象，其中含有的<code>history</code>属性包含了训练过程中损失函数的值以及其他度量指标。

	hist = model.fit(X, y, validation_split=0.2)
	print(hist.history)
</font>
</a>

***

<a name='statful_RNN'>
<font color='000000'>
## 如何使用状态RNN？

一个RNN是状态RNN，意味着训练时每个batch的状态都会被重用于初始化下一个batch的初始状态。

当使用状态RNN时，有如下假设

* 所有的batch都具有相同数目的样本

* 如果<code>X1</code>和<code>X2</code>是两个相邻的batch，那么对于任何<code>i</code>，<code>X2[i]</code>都是<code>X1[i]</code>的后续序列

要使用状态RNN，我们需要

* 显式的指定每个batch的大小。可以通过模型的首层参数<code>batch_input_shape</code>来完成。<code>batch_input_shape</code>是一个整数tuple，例如\(32,10,16\)代表一个具有10个时间步，每步向量长为16，每32个样本构成一个batch的输入数据格式。

* 在RNN层中，设置<code>stateful=True</code>

要重置网络的状态，使用：

* <code>model.reset_states()</code>来重置网络中所有层的状态

* <code>layer.reset_states()</code>来重置指定层的状态

例子：

	X  # this is our input data, of shape (32, 21, 16)
	# we will feed it to our model in sequences of length 10

	model = Sequential()
	model.add(LSTM(32, batch_input_shape=(32, 10, 16), stateful=True))
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

注意，<code>predict</code>，<code>fit</code>，<code>train_on_batch</code>
，<code>predict_classes</code>等方法都会更新模型中状态层的状态。这使得你可以不但可以进行状态网络的训练，也可以进行状态网络的预测。
</font>
</a>