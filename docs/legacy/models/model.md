# 泛型模型接口

为什么叫“泛型模型”，请查看[<font color='#FF0000'>一些基本概念</font>](../getting_started/concepts/#functional)

Keras的泛型模型为```Model```，即广义的拥有输入和输出的模型，我们使用```Model```来初始化一个泛型模型

```python
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(input=a, output=b)
```

在这里，我们的模型以```a```为输入，以```b```为输出，同样我们可以构造拥有多输入和多输出的模型

```python
model = Model(input=[a1, a2], output=[b1, b3, b3])
```

## 常用Model属性

* ```model.layers```：组成模型图的各个层
* ```model.inputs```：模型的输入张量列表
* ```model.outputs```：模型的输出张量列表

***

## Model模型方法

### compile

```python
compile(self, optimizer, loss, metrics=[], loss_weights=None, sample_weight_mode=None)
```
本函数编译模型以供训练，参数有

* optimizer：优化器，为预定义优化器名或优化器对象，参考[<font color='#FF0000'>优化器</font>](../other/optimizers.md)

* loss：目标函数，为预定义损失函数名或一个目标函数，参考[<font color='#FF0000'>目标函数</font>](../other/objectives.md)

* metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法是```metrics=['accuracy']```如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如```metrics={'ouput_a': 'accuracy'}```

* sample_weight_mode：如果你需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。如果模型有多个输出，可以向该参数传入指定sample_weight_mode的字典或列表。在下面```fit```函数的解释中有相关的参考内容。

* kwargs：使用TensorFlow作为后端请忽略该参数，若使用Theano作为后端，kwargs的值将会传递给 K.function

【Tips】如果你只是载入模型并利用其predict，可以不用进行compile。在Keras中，compile主要完成损失函数和优化器的一些配置，是为训练服务的。predict会在内部进行符号函数的编译工作（通过调用_make_predict_function生成函数）【@白菜，@我是小将】
***

### fit

```python
fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
```
本函数用以训练模型，参数有：

* x：输入数据。如果模型只有一个输入，那么x的类型是numpy array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array。如果模型的每个输入都有名字，则可以传入一个字典，将输入名与其输入数据对应起来。

* y：标签，numpy array。如果模型有多个输出，可以传入一个numpy array的list。如果模型的输出拥有名字，则可以传入一个字典，将输出名与其标签对应起来。

* batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。

* nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思

* verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

* callbacks：list，其中的元素是```keras.callbacks.Callback```的对象。这个list中的回调函数将会在训练过程中的适当时机被调用，参考[<font color='#FF0000'>回调函数</font>](../other/callbacks.md)

* validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集。验证集将不参与训练，并在每个epoch结束后测试的模型的指标，如损失函数、精确度等。

* validation_data：形式为（X，y）或（X，y，sample_weights）的tuple，是指定的验证集。此参数将覆盖validation_spilt。

* shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。

* class_weight：字典，将不同的类别映射为不同的权值，该参数用来在训练过程中调整损失函数（只能用于训练）。该参数在处理非平衡的训练数据（某些类的训练样本数很少）时，可以使得损失函数对样本数不足的数据更加关注。

* sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了```sample_weight_mode='temporal'```。

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
fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose=1, callbacks=[], validation_data=None, nb_val_samples=None, class_weight={}, max_q_size=10)
```
利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练

函数的参数是：

* generator：生成器函数，生成器的输出应该为：
	* 一个形如（inputs，targets）的tuple
	
	* 一个形如（inputs, targets,sample_weight）的tuple。所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到```samples_per_epoch```时，记一个epoch结束

* samples_per_epoch：整数，当模型处理的样本达到此数目时计一个epoch结束，执行下一个epoch

* verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录

* validation_data：具有以下三种形式之一
	* 生成验证集的生成器
	
	* 一个形如（inputs,targets）的tuple
	
	* 一个形如（inputs,targets，sample_weights）的tuple
	
* nb_val_samples：仅当```validation_data```是生成器时使用，用以限制在每个epoch结束时用来验证模型的验证集样本数，功能类似于```samples_per_epoch```

* max_q_size：生成器队列的最大容量

函数返回一个```History```对象

例子

```python
def generate_arrays_from_file(path):
    while 1:
    f = open(path)
    for line in f:
        # create numpy arrays of input data
        # and labels, from each line in the file
        x, y = process_line(line)
        yield (x, y)
    f.close()

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
        samples_per_epoch=10000, nb_epoch=10)
```

***

### evaluate_generator
```python
evaluate_generator(self, generator, val_samples, max_q_size=10)
```
本函数使用一个生成器作为数据源，来评估模型，生成器应返回与```test_on_batch```的输入数据相同类型的数据。

函数的参数是：

* generator：生成输入batch数据的生成器

* val_samples：生成器应该返回的总样本数

* max_q_size：生成器队列的最大容量

* nb_worker：使用基于进程的多线程处理时的进程数

* pickle_safe：若设置为True，则使用基于进程的线程。注意因为它的实现依赖于多进程处理，不可传递不可pickle的参数到生成器中，因为它们不能轻易的传递到子进程中。

***

### predict_generator
```python
predict_generator(self, generator, val_samples, max_q_size=10, nb_worker=1, pickle_safe=False)
```
从一个生成器上获取数据并进行预测，生成器应返回与```predict_on_batch```输入类似的数据

函数的参数是：

* generator：生成输入batch数据的生成器

* val_samples：生成器应该返回的总样本数

* max_q_size：生成器队列的最大容量

* nb_worker：使用基于进程的多线程处理时的进程数

* pickle_safe：若设置为True，则使用基于进程的线程。注意因为它的实现依赖于多进程处理，不可传递不可pickle的参数到生成器中，因为它们不能轻易的传递到子进程中。

***

### get_layer
```python
get_layer(self, name=None, index=None)
```
本函数依据模型中层的下标或名字获得层对象，泛型模型中层的下标依据自底向上，水平遍历的顺序。

* name：字符串，层的名字

* index： 整数，层的下标

函数的返回值是层对象