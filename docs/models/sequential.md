# Sequential模型接口

如果刚开始学习Sequential模型，请首先移步[这里](../getting_started/sequential_model.md)阅读文档，本节内容是Sequential的API和参数介绍。

## 常用Sequential属性

* ```model.layers```是添加到模型上的层的list

***

## Sequential模型方法

### add
```python
add(self, layer)
```
向模型中添加一个层

* layer: Layer对象

***

### pop
```python
pop(self)
```
弹出模型最后的一层，无返回值


***

### compile

```python
compile(self, optimizer, loss, metrics=None, sample_weight_mode=None)
```
编译用来配置模型的学习过程，其参数有

* optimizer：字符串（预定义优化器名）或优化器对象，参考[优化器](../other/optimizers.md)

* loss：字符串（预定义损失函数名）或目标函数，参考[损失函数](../other/objectives.md)

* metrics：列表，包含评估模型在训练和测试时的网络性能的指标，典型用法是```metrics=['accuracy']```

* sample_weight_mode：如果你需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。在下面```fit```函数的解释中有相关的参考内容。

* weighted_metrics: metrics列表，在训练和测试过程中，这些metrics将由`sample_weight`或`clss_weight`计算并赋权
* target_tensors: 默认情况下，Keras将为模型的目标创建一个占位符，该占位符在训练过程中将被目标数据代替。如果你想使用自己的目标张量（相应的，Keras将不会在训练时期望为这些目标张量载入外部的numpy数据），你可以通过该参数手动指定。目标张量可以是一个单独的张量（对应于单输出模型），也可以是一个张量列表，或者一个name->tensor的张量字典。

* kwargs：使用TensorFlow作为后端请忽略该参数，若使用Theano/CNTK作为后端，kwargs的值将会传递给 K.function。如果使用TensorFlow为后端，这里的值会被传给tf.Session.run

```python
model = Sequential()
model.add(Dense(32, input_shape=(500,)))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
	  loss='categorical_crossentropy',
	  metrics=['accuracy'])
```

模型在使用前必须编译，否则在调用fit或evaluate时会抛出异常。

### fit

```python
fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
```
本函数将模型训练```nb_epoch```轮，其参数有：

* x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array

* y：标签，numpy array

* batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。

* epochs：整数，训练终止时的epoch值，训练将在达到该epoch值时停止，当没有设置initial_epoch时，它就是训练的总轮数，否则训练的总轮数为epochs - inital_epoch

* verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

* callbacks：list，其中的元素是```keras.callbacks.Callback```的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考[回调函数](../other/callbacks.md)

* validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。注意，validation_split的划分在shuffle之前，因此如果你的数据本身是有序的，需要先手工打乱再指定validation_split，否则可能会出现验证集样本不均匀。

* validation_data：形式为（X，y）的tuple，是指定的验证集。此参数将覆盖validation_spilt。

* shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。

* class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）

* sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了```sample_weight_mode='temporal'```。

* initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。

```fit```函数返回一个```History```的对象，其```History.history```属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况

***
<a name='evaluate'>
<font color='#404040'>
### evaluate
</font>
</a>
```python
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
```
本函数按batch计算在某些输入数据上模型的误差，其参数有：

* x：输入数据，与```fit```一样，是numpy array或numpy array的list

* y：标签，numpy array

* batch_size：整数，含义同```fit```的同名参数

* verbose：含义同```fit```的同名参数，但只能取0或1

* sample_weight：numpy array，含义同```fit```的同名参数

本函数返回一个测试误差的标量值（如果模型没有其他评价指标），或一个标量的list（如果模型还有其他的评价指标）。```model.metrics_names```将给出list中各个值的含义。

如果没有特殊说明，以下函数的参数均保持与```fit```的同名参数相同的含义

如果没有特殊说明，以下函数的verbose参数（如果有）均只能取0或1


***

### predict

```python
predict(self, x, batch_size=32, verbose=0)
```
本函数按batch获得输入数据对应的输出，其参数有：

函数的返回值是预测值的numpy array

***

### train_on_batch
```python
train_on_batch(self, x, y, class_weight=None, sample_weight=None)
```
本函数在一个batch的数据上进行一次参数更新

函数返回训练误差的标量值或标量值的list，与[evaluate](#evaluate)的情形相同。

***

### test_on_batch
```python
test_on_batch(self, x, y, sample_weight=None)
```
本函数在一个batch的样本上对模型进行评估

函数的返回与[evaluate](#evaluate)的情形相同

***

### predict_on_batch
```python
predict_on_batch(self, x)
```
本函数在一个batch的样本上对模型进行测试

函数返回模型在一个batch上的预测结果

***

### fit_generator
```python
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)
```
利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练

函数的参数是：

* generator：生成器函数，生成器的输出应该为：
	* 一个形如（inputs，targets）的tuple
	
	* 一个形如（inputs, targets,sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到```samples_per_epoch```时，记一个epoch结束

* steps_per_epoch：整数，当生成器返回```steps_per_epoch```次数据时计一个epoch结束，执行下一个epoch

* epochs：整数，数据迭代的轮数

* verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

* validation_data：具有以下三种形式之一
	* 生成验证集的生成器
	
	* 一个形如（inputs,targets）的tuple
	
	* 一个形如（inputs,targets，sample_weights）的tuple

* validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数
	
* class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。

* sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了```sample_weight_mode='temporal'```。

* workers：最大进程数

* max_q_size：生成器队列的最大容量

* pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递non picklable（无法被pickle序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。

* initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。


函数返回一个```History```对象

例子：

```python
def generate_arrays_from_file(path):
    while 1:
    		f = open(path)
    		for line in f:
        		# create Numpy arrays of input data
        		# and labels, from each line in the file
        		x, y = process_line(line)
        		yield (x, y)
    	f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, epochs=10)
```

***

### evaluate_generator
```python
evaluate_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False)
```
本函数使用一个生成器作为数据源评估模型，生成器应返回与```test_on_batch```的输入数据相同类型的数据。该函数的参数与```fit_generator```同名参数含义相同，steps是生成器要返回数据的轮数。

***

### predict_generator
```python
predict_generator(self, generator, steps, max_q_size=10, workers=1, pickle_safe=False, verbose=0)
```
本函数使用一个生成器作为数据源预测模型，生成器应返回与```test_on_batch```的输入数据相同类型的数据。该函数的参数与```fit_generator```同名参数含义相同，steps是生成器要返回数据的轮数。

***
