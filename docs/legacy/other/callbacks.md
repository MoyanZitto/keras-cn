# 回调函数Callbacks

回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。通过传递回调函数列表到模型的```.fit()```中，即可在给定的训练阶段调用该函数集中的函数。

【Tips】虽然我们称之为回调“函数”，但事实上Keras的回调函数是一个类，回调函数只是习惯性称呼

## CallbackList
```python
keras.callbacks.CallbackList(callbacks=[], queue_length=10)
```

## Callback
```python
keras.callbacks.Callback()
```
这是回调函数的抽象类，定义新的回调函数必须继承自该类

### 类属性

* params：字典，训练参数集（如信息显示方法verbosity，batch大小，epoch数）

* model：```keras.models.Model```对象，为正在训练的模型的引用

回调函数以字典```logs```为参数，该字典包含了一系列与当前batch或epoch相关的信息。

目前，模型的```.fit()```中有下列参数会被记录到```logs```中：

* 在每个epoch的结尾处（on_epoch_end），```logs```将包含训练的正确率和误差，```acc```和```loss```，如果指定了验证集，还会包含验证集正确率和误差```val_acc)```和```val_loss```，```val_acc```还额外需要在```.compile```中启用```metrics=['accuracy']```。

* 在每个batch的开始处（on_batch_begin）：```logs```包含```size```，即当前batch的样本数

* 在每个batch的结尾处（on_batch_end）：```logs```包含```loss```，若启用```accuracy```则还包含```acc```

***

## BaseLogger
```python
keras.callbacks.BaseLogger()
```
该回调函数用来对每个epoch累加```metrics```指定的监视指标的epoch平均值

该回调函数在每个Keras模型中都会被自动调用

***

## ProgbarLogger
```python
keras.callbacks.ProgbarLogger()
```
该回调函数用来将```metrics```指定的监视指标输出到标准输出上

***

## History
```python
keras.callbacks.History()
```
该回调函数在Keras模型上会被自动调用，```History```对象即为```fit```方法的返回值

***

## ModelCheckpoint
```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```
该回调函数将在每个epoch后保存模型到```filepath```

```filepath```可以是格式化的字符串，里面的占位符将会被```epoch```值和传入```on_epoch_end```的```logs```关键字所填入

例如，```filepath```若为```weights.{epoch:02d-{val_loss:.2f}}.hdf5```，则会生成对应epoch和验证集loss的多个文件。

### 参数

* filename：字符串，保存模型的路径

* monitor：需要监视的值

* verbose：信息展示模式，0或1

* save_best_only：当设置为```True```时，将只保存在验证集上性能最好的模型

* mode：‘auto’，‘min’，‘max’之一，在```save_best_only=True```时决定性能最佳模型的评判准则，例如，当监测值为```val_acc```时，模式应为```max```，当检测值为```val_loss```时，模式应为```min```。在```auto```模式下，评价准则由被监测值的名字自动推断。

* save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）

* period：CheckPoint之间的间隔的epoch数

***

## EarlyStopping
```python
keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
```
当监测值不再改善时，该回调函数将中止训练

### 参数

* monitor：需要监视的量

* patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过```patience```个epoch后停止训练。

* verbose：信息展示模式

* mode：‘auto’，‘min’，‘max’之一，在```min```模式下，如果检测值停止下降则中止训练。在```max```模式下，当检测值不再上升则停止训练。

***

## RemoteMonitor
```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000')
```
该回调函数用于向服务器发送事件流，该回调函数需要```requests```库

### 参数

* root：该参数为根url，回调函数将在每个epoch后把产生的事件流发送到该地址，事件将被发往```root + '/publish/epoch/end/'```。发送方法为HTTP POST，其```data```字段的数据是按JSON格式编码的事件字典。

***

## LearningRateScheduler
```python
keras.callbacks.LearningRateScheduler(schedule)
```
该回调函数是学习率调度器

### 参数

* schedule：函数，该函数以epoch号为参数（从0算起的整数），返回一个新学习率（浮点数）

***

## TensorBoard
```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)
```
该回调函数是一个可视化的展示器

TensorBoard是TensorFlow提供的可视化工具，该回调函数将日志信息写入TensorBorad，使得你可以动态的观察训练和测试指标的图像以及不同层的激活值直方图。

如果已经通过pip安装了TensorFlow，我们可通过下面的命令启动TensorBoard：

```python
tensorboard --logdir=/full_path_to_your_logs
```
更多的参考信息，请点击[<font color='#FF0000'>这里</font>](http://keras.io/https__://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html)

### 参数

* log_dir：保存日志文件的地址，该文件将被TensorBoard解析以用于可视化

* histogram_freq：计算各个层激活值直方图的频率（每多少个epoch计算一次），如果设置为0则不计算。

***

## ReduceLROnPlateau
```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```
当评价指标不在提升时，减少学习率

当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在`patience`个epoch中看不到模型性能提升，则减少学习率

### 参数

- monitor：被监测的量
- factor：每次减少学习率的因子，学习率将以`lr = lr*factor`的形式被减少
- patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
- mode：‘auto’，‘min’，‘max’之一，在```min```模式下，如果检测值触发学习率减少。在```max```模式下，当检测值不再上升则触发学习率减少。
- epsilon：阈值，用来确定是否进入检测值的“平原区”
- cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
- min_lr：学习率的下限

### 示例：
```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
            patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
``` 

##CSVLogger
```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```
将epoch的训练结果保存在csv文件中，支持所有可被转换为string的值，包括1D的可迭代数值如np.ndarray.

###参数

- fiename：保存的csv文件名，如`run/log.csv`
- separator：字符串，csv分隔符
- append：默认为False，为True时csv文件如果存在则继续写入，为False时总是覆盖csv文件

### 示例

```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```

## LambdaCallback
```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```
用于创建简单的callback的callback类

该callback的匿名函数将会在适当的时候调用，注意，该回调函数假定了一些位置参数`on_eopoch_begin`和`on_epoch_end`假定输入的参数是`epoch, logs`. `on_batch_begin`和`on_batch_end`假定输入的参数是`batch, logs`，`on_train_begin`和`on_train_end`假定输入的参数是`logs`

### 参数

- on_epoch_begin: 在每个epoch开始时调用
- on_epoch_end: 在每个epoch结束时调用
- on_batch_begin: 在每个batch开始时调用
- on_batch_end: 在每个batch结束时调用
- on_train_begin: 在训练开始时调用
- on_train_end: 在训练结束时调用

### 示例

```python
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(on_batch_begin=lambda batch, logs: print(batch))

# Plot the loss after every epoch.
import numpy as np
import matplotlib.pyplot as plt
plot_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch), logs['loss']))

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(on_train_end=lambda logs: [p.terminate() for p in processes if p.is_alive()])

model.fit(..., callbacks=[batch_print_callback, plot_loss_callback, cleanup_callback])
```

## 编写自己的回调函数

我们可以通过继承```keras.callbacks.Callback```编写自己的回调函数，回调函数通过类成员```self.model```访问访问，该成员是模型的一个引用。

这里是一个简单的保存每个batch的loss的回调函数：

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

### 例子：记录损失函数的历史数据
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, callbacks=[history])

print history.losses
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
```

### 例子：模型检查点
```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
```


