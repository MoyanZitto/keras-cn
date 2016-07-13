# Keras后端

## 什么是“后端”

Keras是一个模型级的库，提供了快速构建深度学习网络的模块。Keras并不处理如张量乘法、卷积等底层操作。这些操作依赖于某种特定的、优化良好的张量操作库。Keras依赖于处理张量的库就称为“后端引擎”。Keras提供了两种后端引擎Theano/Tensorflow，并将其函数统一封装，使得用户可以以同一个接口调用不同后端引擎的函数。

* Theano是一个开源的符号主义张量操作框架，由蒙特利尔大学LISA/MILA实验室开发

* TensorFlow是一个符号主义的张量操作框架，由Google开发。

## 切换后端

如果你至少运行过一次Keras，你将在下面的目录下找到Keras的配置文件：

```~/.keras/keras.json```

如果该目录下没有该文件，你可以手动创建一个。

文件的内容大概如下：

```{"epsilon": 1e-07, "floatx": "float32", "backend": "theano"}```

将```backend```字段的值改写为你需要使用的后端：```theano```或```tensorflow```，即可完成后端的切换。

我们也可以通过定义环境变量```KERAS_BACKEND```来覆盖上面配置文件中定义的后端：

```python
KERAS_BACKEND=tensorflow python -c "from keras import backend; print backend._BACKEND"
Using TensorFlow backend.
tensorflow
```

## 使用抽象的Keras后端来编写代码

如果你希望你编写的Keras模块能够同时在Theano和TensorFlow两个后端上使用，你可以通过Keras后端接口来编写代码，这里是一个简介：

```python
from keras import backend as K
```
下面的代码实例化了一个输入占位符，等价于```tf.placeholder()``` ，```T.matrix()```，```T.tensor3()```等

```python
input = K.placeholder(shape=(2, 4, 5))
# also works:
input = K.placeholder(shape=(None, 4, 5))
# also works:
input = K.placeholder(ndim=3)
```
下面的代码实例化了一个共享变量（shared），等价于```tf.variable()```或 ```theano.shared()```.

```python
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# all-ones:
var = K.ones(shape=(3, 4, 5))
```

大多数你需要的张量操作都可以通过统一的Keras后端接口完成，而不关心具体执行这些操作的是Theano还是TensorFlow
```python
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=2)
a = K.softmax(b)
a = concatenate([b, c], axis=-1)
# etc...
```

## Kera后端函数

###epsilon
```python
epsilon()
```

以数值形式返回一个（一般来说很小的）数，即fuzz factor

###set_epsilon
```python
set_epsilon()
```

设置在数值表达式中使用的fuzz factor

### learning_phase

```python
learning_phase()
```
返回训练模式/测试模式的flag，该flag是一个用以传入Keras模型的标记，以决定当前模型执行于训练模式下还是测试模式下。

### floatx
```python
floatx()
```
返回默认的浮点数数据类型，为字符串，如 'float16', 'float32', 'float64'.

### cast_to_floatx
```python
cast_to_floatx(x)
```
将numpy array转换为floatx

### image_dim_ordering
```python
image_dim_ordering()
```
返回图像的维度顺序（‘tf’或‘th’）

### set_image_dim_ordering
```python
set_image_dim_ordering()
```
设置图像的维度顺序（‘tf’或‘th’）


### shape
```python
shape(x)
```
返回一个张量的符号shape

### variable
```python
variable(value, dtype='float32', name=None)
```
实例化一个张量，返回之

* value：用来初始化张量的值

* dtype：张量数据类型

* name：张量的名字（可选）

### placeholder
```python
placeholder(shape=None, ndim=None, dtype='float32', name=None)
```
实例化一个占位符，返回之

* shape：占位符的shape（整数tuple，可能包含None） 

* ndim: 占位符张量的阶数，要初始化一个占位符，至少指定```shape```和```ndim```之一，如果都指定则使用```shape```

* dtype: 占位符数据类型

* name: 占位符名称（可选）

### int_shape
```python
int_shape(x)
```
以整数Tuple或None的形式返回张量shape

### ndim
```python
ndim(x)
```
返回张量的阶数，为整数

### dtype
```python
dtype(x)
```
返回张量的数据类型，为字符串

### eval
```python
eval(x)
```
求得张量的值，返回一个Numpy array

### zeros
```python
zeros(shape, dtype='float32', name=None)
```
生成一个全0张量


### ones
```python
ones(shape, dtype='float32', name=None)
```
生成一个全1张量

### eye
```python
eye(size, dtype='float32', name=None)
```
生成一个单位矩阵

### zeros_like
```python
zeros_like(x, name=None)
```
生成与另一个张量shape相同的全0张量

### ones_like
```python
ones_like(x, name=None)
```
生成与另一个张量shape相同的全1张量

### count_params
```python
count_params(x)
```
返回张量中标量的个数

###cast
```python
cast(x, dtype)
```
改变张量的数据类型

### dot
```python
dot(x, y)
```
求两个张量的乘积。当试图计算两个N阶张量的乘积时，与Theano行为相同，如```(2, 3).(4, 3, 5) = (2, 4, 5))```

### batch_dot
```python
batch_dot(x, y, axes=None)
```
按批进行张量乘法，该函数将产生比输入张量维度低的张量，如果张量的维度被减至1，则通过```expand_dims```保证其维度至少为2

例如，假设```x = [[1, 2],[3,4]]``` ， ```y = [[5, 6],[7, 8]]```，则``` batch_dot(x, y, axes=1) = [[17, 53]] ```，即```x.dot(y.T)```的主对角元素，此过程中我们没有计算过反对角元素的值。 

* x,y：阶数大于等于2的张量

* axes：目标结果的维度，为整数或整数列表。

### transpose
```python
transpose(x)
```
矩阵转置

### gather
```python
gather(reference, indices)
```
在给定的2D张量中检索给定下标的向量
Retrieves the vectors of indices indices in the 2D tensor reference.

* reference：2D张量

* indices：整数张量，其元素为要查询的下标

返回一个与```reference```数据类型相同的3D张量

### max
```python
max(x, axis=None, keepdims=False)
```
求张量中的最大值

###min
```python
min(x, axis=None, keepdims=False)
```
求张量中的最小值

###sum
```python
sum(x, axis=None, keepdims=False)
```
在给定轴上计算张量中元素之和

### prod
```python
prod(x, axis=None, keepdims=False)
```
在给定轴上计算张量中元素之积

### std
```python
std(x, axis=None, keepdims=False)
```
在给定轴上求张量元素之标准差

### mean
```python
mean(x, axis=None, keepdims=False)
```
在给定轴上求张量元素之均值

### any
```python
any(x, axis=None, keepdims=False)
```
按位或，返回数据类型为uint8的张量（元素为0或1）

### all
```python
any(x, axis=None, keepdims=False)
```
按位与，返回类型为uint8de tensor

### argmax
```python
argmax(x, axis=-1)
```
在给定轴上求张量之最大元素下标

### argmin
```python
argmin(x, axis=-1)
```
在给定轴上求张量之最小元素下标

###square
```python
square(x)
```
逐元素平方

### abs
```python
abs(x)
```
逐元素绝对值

###sqrt
```python
sqrt(x)
```
逐元素开方

###exp
```python
exp(x)
```
逐元素求自然指数

###log
```python
log(x)
```
逐元素求自然对数

###round
```python
round(x)
```
逐元素四舍五入

###sign
```python
sign(x)
```
逐元素求元素的符号（+1或-1）

###pow
```python
pow(x, a)
```
逐元素求x的a次方

###clip
```python
clip(x, min_value, max_value)
```
逐元素clip（将超出指定范围的数强制变为边界值）

###equal
```python
equal(x, y)
```
逐元素判相等关系，返回布尔张量

###not_equal
```python
not_equal(x, y)
```
逐元素判不等关系，返回布尔张量

###maximum
```python
maximum(x, y)
```
逐元素取两个张量的最大值

###minimum
```python
minimum(x, y)
```
逐元素取两个张量的最小值

###sin
```python
sin(x)
```
逐元素求正弦值

###cos
```python
cos(x)
```
逐元素求余弦值

###concatenate
```python
concatenate(tensors, axis=-1)
```
在给定轴上将一个列表中的张量串联为一个张量 specified axis.

###reshape
```python
reshape(x, shape)
```
将张量的shape变换为指定shape

###permute_dimensions
```python
permute_dimensions(x, pattern)
```
按照给定的模式重排一个张量的轴

* pattern:：代表维度下标的tuple如```(0, 2, 1)```.

###resize_images
```python
resize_images(X, height_factor, width_factor, dim_ordering)
```
依据给定的缩放因子，改变一个batch图片的大小，参数中的两个因子都为正整数，图片的排列顺序与维度的模式相关，如‘th’和‘tf’


###repeat_elements
```python
repeat_elements(x, rep, axis)
```
在给定轴上重复张量元素```rep```次，与```np.repeat```类似。例如，若xshape```(s1, s2, s3) ```并且给定轴为```axis=1```，输出张量的shape为```(s1, s2 * rep, s3)```

###repeat
```python
repeat(x, n)
```
重复2D张量，例如若xshape是```(samples, dim)```且n为2，则输出张量的shape是```(samples, 2, dim)```

###batch_flatten
```python
batch_flatten(x)
```
将一个n阶张量转变为2阶张量，其第一维度保留不变

###expand_dims
```python
expand_dims(x, dim=-1)
```
在下标为```dim```的轴上增加一维

###squeeze
```python
squeeze(x, axis)
```
将下标为```axis```的一维从张量中移除

###temporal_padding
```python
temporal_padding(x, padding=1)
```
向3D张量中间的那个维度的左右两端填充```padding```个0值

###spatial_2d_padding
```python
spatial_2d_padding(x, padding=(1, 1), dim_ordering='th')
```
向4D张量第二和第三维度的左右两端填充```padding[0]```和```padding[1]```个0值.

###get_value
```python
get_value(x)
```
以Numpy array的形式返回张量的值

###set_value
```python
set_value(x, value)
```
从numpy array将值载入张量中

###batch_set_value
```python
batch_set_value(tuples)
```
将多个值载入多个张量变量中

* tuples: 列表，其中的元素形如```(tensor, value)```。```value```是要载入的Numpy array数据

###function
```python
function(inputs, outputs, updates=[])
```
实例化一个Keras函数

* inputs:：列表，其元素为占位符或张量变量

* outputs：输出张量的列表

* updates：列表，其元素是形如```(old_tensor, new_tensor)```的tuple.

###gradients
```python
gradients(loss, variables)
```
返回loss函数关于variables的梯度，variables为张量变量的列表

###rnn
```python
rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```
在张量的时间维上迭代

* inputs： 形如```(samples, time, ...) ```的时域信号的张量，阶数至少为3

* step_function：每个时间步要执行的函数，其参数有

	* input：形如```(samples, ...)```的张量，不含时间维，代表某个时间步时一个batch的样本
	
	* states：张量列表
	
	* step_functions的返回两个值，为：
	
		* output：形如```(samples, ...)```的张量
		
		* new_states：张量列表，与‘states’的长度相同
		
* initial_states：形如```(samples, ...)```的张量，包含了```step_function```状态的初始值。

* go_backwards：布尔值，若设为True，则逆向迭代序列

* mask：形如```(samples, time, 1) ```的二值张量，需要屏蔽的数据元素上值为1

* constants：按时间步传递给函数的常数列表

* unroll：当使用TensorFlow时，RNN总是展开的。当使用Theano时，设置该值为```True```将展开递归网络

* input_length：使用TensorFlow时不需要此值，在使用Theano时，如果要展开递归网络，必须指定输入序列

函数的返回值是形如```(last_output, outputs, new_states)```的tuple

* last_output：rnn最后的输出，形如```(samples, ...)```

* outputs：形如```(samples, time, ...) ```的张量，每个在\[s,t\]点的输出对应于样本s在t时间的输出。

* new_states: 列表，其元素为形如```(samples, ...)```的张量，代表每个样本的最后一个状态。

###switch
```python
switch(condition, then_expression, else_expression)
```
依据给定的条件‘condition’（整数或布尔值）在两个表达式之间切换，注意两个表达式都应该是具有同样shape的符号化张量表达式

* condition：标量张量
* then_expression：TensorFlow表达式
* else_expression: TensorFlow表达式

###in_train_phase
```python
in_train_phase(x, alt)
```
如果处于训练模式，则选择x，否则选择alt，注意alt应该与x的shape相同

###in_test_phase
```python
in_test_phase(x, alt)
```
如果处于测试模式，则选择x，否则选择alt，注意alt应该与x的shape相同

###relu
```python
relu(x, alpha=0.0, max_value=None)
```
修正线性单元

* alpha：负半区斜率
* max_value: 饱和门限

### softmax
```python
softmax(x)
```
返回张量的softmax值

###softplus
```python
softplus(x)
```
返回张量的softplus值

###categorical_crossentropy
```python
categorical_crossentropy(output, target, from_logits=False)
```
计算输出张量和目标张量的Categorical crossentropy（类别交叉熵），目标张量与输出张量必须shape相同

###sparse_categorical_crossentropy
```python
sparse_categorical_crossentropy(output, target, from_logits=False)
```python
计算输出张量和目标张量的Categorical crossentropy（类别交叉熵），目标张量必须是整型张量

###binary_crossentropy
```python
binary_crossentropy(output, target, from_logits=False)
```
计算输出张量和目标张量的交叉熵

###sigmoid
```python
sigmoid(x)
```
逐元素计算sigmoid值

###hard_sigmoid
```python
hard_sigmoid(x)
```
该函数是分段线性近似的sigmoid，计算速度更快

###tanh
```python
tanh(x)
```
逐元素计算sigmoid值

###dropout
```python
dropout(x, level, seed=None)
```
随机将x中一定比例的值设置为0，并放缩整个tensor

Arguments

* x：张量

* level：x中设置成0的元素比例

* seed：随机数种子

###l2_normalize
```python
l2_normalize(x, axis)
```
在给定轴上对张量进行L2范数规范化

###conv2d
```python
conv2d(x, kernel, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None)
```
2D卷积

参数：

* kernel：卷积核张量

* strides：步长，长为2的tuple

* border_mode：“same”，“valid”之一的字符串

* dim_ordering：“tf”和“th”之一，维度排列顺序

### deconv2d
```python
deconv2d(x, kernel, output_shape, strides=(1, 1), border_mode='valid', dim_ordering='th', image_shape=None, filter_shape=None)
```
2D反卷积（转置卷积）

参数：

* x：输入张量

* kernel：卷积核张量

* output_shape: 输出shape的1D的整数张量

* strides：步长，tuple类型

* border_mode：‘same’或‘valid’

* dim_ordering：‘tf’或‘th’

### conv3d
```python
conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid', dim_ordering='th', volume_shape=None, filter_shape=None)
```
3D卷积

参数：

* x：输入张量

* kernel：卷积核张量

* output_shape: 输出shape的1D的整数张量

* strides：步长，tuple类型

* border_mode：‘same’或‘valid’

* dim_ordering：‘tf’或‘th’

### pool2d
```python
pool2d(x, pool_size, strides=(1, 1), border_mode='valid', dim_ordering='th', pool_mode='max')
```
2D池化

* pool_size：含有两个整数的tuple，池的大小

* strides: 含有两个整数的tuple，步长

* border_mode：“same”，“valid”之一的字符串

* dim_ordering：“tf”和“th”之一，维度排列顺序

* pool_mode: “max”，“avg”之一，池化方式

### pool3d
```python
pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid', dim_ordering='th', pool_mode='max')
```
3D池化

* pool_size：含有3个整数的tuple，池的大小

* strides: 含有3个整数的tuple，步长

* border_mode：“same”，“valid”之一的字符串

* dim_ordering：“tf”和“th”之一，维度排列顺序

* pool_mode: “max”，“avg”之一，池化方式