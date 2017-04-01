# （批）规范化BatchNormalization

## BatchNormalization层
```python
keras.layers.normalization.BatchNormalization(epsilon=1e-06, mode=0, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')
```
该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1

### 参数

* epsilon：大于0的小浮点数，用于防止除0错误

* mode：整数，指定规范化的模式，取0或1
	
	* 0：按特征规范化，输入的各个特征图将独立被规范化。规范化的轴由参数```axis```指定。注意，如果输入是‘th’格式形状的（samples，channels，rows，cols）的4D图像张量，则应设置规范化的轴为1，即沿着通道轴规范化。在训练阶段，我们使用每个batch数据的统计信息（如：均值、标准差等）来对训练数据进行规范化，而在测试阶段，我们使用训练时得到的统计信息的滑动平均来对测试数据进行规范化。	
	* 1：按样本规范化，该模式默认输入为2D
	* 2: 按特征规范化, 与模式0相似, 但不同之处在于：在训练和测试过程中，均使用每个batch数据的统计信息而分别对训练数据和测试数据进行规范化。
* axis：整数，指定当```mode=0```时规范化的轴。例如输入是形如（samples，channels，rows，cols）的4D图像张量，则应设置规范化的轴为1，意味着对每个特征图进行规范化

* momentum：在按特征规范化时，计算数据的指数平均数和标准差时的动量

* weights：初始化权重，为包含2个numpy array的list，其shape为```[(input_shape,),(input_shape)]```

* beta_init：beta的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* gamma_init：gamma的初始化方法，为预定义初始化方法名的字符串，或用于初始化权重的Theano函数。该参数仅在不传递```weights```参数时有意义。

* gamma_regularizer：WeightRegularizer(如L1或L2正则化)的实例，作用在gamma向量上。

* beta_regularizer：WeightRegularizer的实例，作用在beta向量上。


### 输入shape

任意，当使用本层为模型首层时，指定```input_shape```参数时有意义。

### 输出shape

与输入shape相同

### 参考文献

* [<font color='FF0000'>Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift</font>](http://arxiv.org/pdf/1502.03167v3.pdf)

【Tips】统计学习的一个重要假设是源空间与目标空间的数据分布是一致的，而神经网络各层输出的分布不一定与输入一致，尤其当网络越深，这种不一致越明显。BatchNormalization把分布一致弱化为均值与方差一致，然而即使是这种弱化的版本也对学习过程起到了重要效果。另一方面，BN的更重要作用是防止梯度弥散，它通过将激活值规范为统一的均值和方差，将原本会减小的激活值得到放大。【@Bigmoyan】

