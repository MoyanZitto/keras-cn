# Keras后端

## 什么是“后端”

Keras是一个模型级的库，提供了快速构建深度学习网络的模块。Keras并不处理如张量乘法、卷积等底层操作。这些操作依赖于某种特定的、优化良好的张量操作库。Keras依赖于处理张量的库就称为“后端引擎”。Keras提供了两种后端引擎Theano/Tensorflow，并将其函数统一封装，使得用户可以以同一个接口调用不同后端引擎的函数

* Theano是一个开源的符号主义张量操作框架，由蒙特利尔大学LISA/MILA实验室开发

* TensorFlow是一个符号主义的张量操作框架，由Google开发

在未来，我们有可能要添加更多的后端选项，如果你有兴趣开发后端，请与我联系~

## 切换后端

如果你至少运行过一次Keras，你将在下面的目录下找到Keras的配置文件：

```~/.keras/keras.json```

如果该目录下没有该文件，你可以手动创建一个

文件的默认配置如下：

```
{
"image_dim_ordering":"tf",
"epsilon":1e-07,
"floatx":"float32",
"backend":"tensorflow"
}
```

将```backend```字段的值改写为你需要使用的后端：```theano```或```tensorflow```，即可完成后端的切换

我们也可以通过定义环境变量```KERAS_BACKEND```来覆盖上面配置文件中定义的后端：

```python
KERAS_BACKEND=tensorflow python -c "from keras import backend;"
Using TensorFlow backend.
```

## keras.json 细节
```python
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```
你可以更改以上`~/.keras/keras.json`中的配置

- `image_dim_ordering`：字符串，"tf"或"th"，该选项指定了Keras将要使用的维度顺序，可通过`keras.backend.image_dim_ordering()`来获取当前的维度顺序。对2D数据来说，`tf`假定维度顺序为(rows,cols,channels)而`th`假定维度顺序为(channels, rows, cols)。对3D数据而言，`tf`假定(conv_dim1, conv_dim2, conv_dim3, channels)，`th`则是(channels, conv_dim1, conv_dim2, conv_dim3)

- `epsilon`：浮点数，防止除0错误的小数字
- `floatx`：字符串，"float16", "float32", "float64"之一，为浮点数精度
- `backend`：字符串，所使用的后端，为"tensorflow"或"theano"



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
下面的代码实例化了一个共享变量（shared），等价于```tf.variable()```或 ```theano.shared()```

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

以数值形式返回一个（一般来说很小的）数，用以防止除0错误

###set_epsilon
```python
set_epsilon(e)
```

设置在数值表达式中使用的fuzz factor，用于防止除0错误，该值应该是一个较小的浮点数，示例：
```python
>>> from keras import backend as K
>>> K.epsilon()
1e-08
>>> K.set_epsilon(1e-05)
>>> K.epsilon()
1e-05
```

### floatx
```python
floatx()
```
返回默认的浮点数数据类型，为字符串，如 'float16', 'float32', 'float64'

### set_floatx(floatx)
```python
floatx()
```
设置默认的浮点数数据类型，为字符串，如 'float16', 'float32', 'float64',示例：
```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> K.set_floatx('float16')
>>> K.floatx()
'float16'
```


### cast_to_floatx
```python
cast_to_floatx(x)
```
将numpy array转换为默认的Keras floatx类型，x为numpy array，返回值也为numpy array但其数据类型变为floatx。示例：
```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> arr = numpy.array([1.0, 2.0], dtype='float64')
>>> arr.dtype
dtype('float64')
>>> new_arr = K.cast_to_floatx(arr)
>>> new_arr
array([ 1.,  2.], dtype=float32)
>>> new_arr.dtype
dtype('float32')
```

### image_dim_ordering
```python
image_dim_ordering()
```
返回默认的图像的维度顺序（‘tf’或‘th’）

### set_image_dim_ordering
```python
set_image_dim_ordering(dim_ordering)
```
设置图像的维度顺序（‘tf’或‘th’）,示例：
```python
>>> from keras import backend as K
>>> K.image_dim_ordering()
'th'
>>> K.set_image_dim_ordering('tf')
>>> K.image_dim_ordering()
'tf'
```

### get_uid
```python
get_uid(prefix='')
```
依据给定的前缀提供一个唯一的UID，参数为表示前缀的字符串，返回值为整数，示例：
```python
>>> keras.backend.get_uid('dense')
>>> 1
>>> keras.backend.get_uid('dense')
>>> 2
```

### is_keras_tensor
```python
is_keras_tensor(x)
```

判断x是否是一个Keras tensor，返回一个布尔值，示例
```python
>>> from keras import backend as K
>>> np_var = numpy.array([1, 2])
>>> K.is_keras_tensor(np_var)
False
>>> keras_var = K.variable(np_var)
>>> K.is_keras_tensor(keras_var)  # A variable is not a Tensor.
False
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> K.is_keras_tensor(keras_placeholder)  # A placeholder is a Tensor.
True
```

### clear_session
```python
clear_session()
```
结束当前的TF计算图，并新建一个。有效的避免模型/层的混乱

### manual_variable_initialization
```python
manual_variable_initialization(value)
```
指出变量应该以其默认值被初始化还是由用户手动初始化，参数value为布尔值，默认False代表变量由其默认值初始化

### learning_phase

```python
learning_phase()
```
返回训练模式/测试模式的flag，该flag是一个用以传入Keras模型的标记，以决定当前模型执行于训练模式下还是测试模式下

### set_learning_phase

```python
set_learning_phase()
```
设置训练模式/测试模式0或1

### is_sparse
```python
is_sparse(tensor)
```
判断一个tensor是不是一个稀疏的tensor(稀不稀疏由tensor的类型决定，而不是tensor实际上有多稀疏)，返回值是一个布尔值，示例：
```python
>>> from keras import backend as K
>>> a = K.placeholder((2, 2), sparse=False)
>>> print(K.is_sparse(a))
False
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
```

### to_dense
```python
to_dense(tensor)
```
将一个稀疏tensor转换一个不稀疏的tensor并返回之，示例：
```python
>>> from keras import backend as K
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
>>> c = K.to_dense(b)
>>> print(K.is_sparse(c))
False
```

### variable
```python
variable(value, dtype='float32', name=None)
```
实例化一个张量，返回之

参数：

* value：用来初始化张量的值
* dtype：张量数据类型
* name：张量的名字（可选）

示例：
```python
>>> from keras import backend as K
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val, dtype='float64', name='example_var')
>>> K.dtype(kvar)
'float64'
>>> print(kvar)
example_var
>>> kvar.eval()
array([[ 1.,  2.],
   [ 3.,  4.]])
```

### placeholder
```python
placeholder(shape=None, ndim=None, dtype='float32', name=None)
```
实例化一个占位符，返回之

参数：

* shape：占位符的shape（整数tuple，可能包含None） 
* ndim: 占位符张量的阶数，要初始化一个占位符，至少指定```shape```和```ndim```之一，如果都指定则使用```shape```
* dtype: 占位符数据类型
* name: 占位符名称（可选）

示例：
```python
>>> from keras import backend as K
>>> input_ph = K.placeholder(shape=(2, 4, 5))
>>> input_ph._keras_shape
(2, 4, 5)
>>> input_ph
<tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
```

### shape
```python
shape(x)
```
返回一个张量的符号shape，符号shape的意思是返回值本身也是一个tensor，示例：
```python
>>> from keras import backend as K
>>> tf_session = K.get_session()
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> input = keras.backend.placeholder(shape=(2, 4, 5))
>>> K.shape(kvar)
<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
>>> K.shape(input)
<tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
__To get integer shape (Instead, you can use K.int_shape(x))__

>>> K.shape(kvar).eval(session=tf_session)
array([2, 2], dtype=int32)
>>> K.shape(input).eval(session=tf_session)
array([2, 4, 5], dtype=int32)
```

### int_shape
```python
int_shape(x)
```
以整数Tuple或None的形式返回张量shape，示例：
```python
>>> from keras import backend as K
>>> input = K.placeholder(shape=(2, 4, 5))
>>> K.int_shape(input)
(2, 4, 5)
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.int_shape(kvar)
(2, 2)
```

### ndim
```python
ndim(x)
```
返回张量的阶数，为整数，示例：
```python
>>> from keras import backend as K
>>> input = K.placeholder(shape=(2, 4, 5))
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.ndim(input)
3
>>> K.ndim(kvar)
2
```

### dtype
```python
dtype(x)
```
返回张量的数据类型，为字符串，示例：
```python
>>> from keras import backend as K
>>> K.dtype(K.placeholder(shape=(2,4,5)))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
'float64'
__Keras variable__

>>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
>>> K.dtype(kvar)
'float32_ref'
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.dtype(kvar)
'float32_ref'
```

### eval
```python
eval(x)
```
求得张量的值，返回一个Numpy array，示例：
```python
>>> from keras import backend as K
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.eval(kvar)
array([[ 1.,  2.],
   [ 3.,  4.]], dtype=float32)
```

### zeros
```python
zeros(shape, dtype='float32', name=None)
```
生成一个全0张量，示例：
```python
>>> from keras import backend as K
>>> kvar = K.zeros((3,4))
>>> K.eval(kvar)
array([[ 0.,  0.,  0.,  0.],
   [ 0.,  0.,  0.,  0.],
   [ 0.,  0.,  0.,  0.]], dtype=float32)
```

### ones
```python
ones(shape, dtype='float32', name=None)
```
生成一个全1张量，示例
```python
>>> from keras import backend as K
>>> kvar = K.ones((3,4))
>>> K.eval(kvar)
array([[ 1.,  1.,  1.,  1.],
   [ 1.,  1.,  1.,  1.],
   [ 1.,  1.,  1.,  1.]], dtype=float32)
```

### eye
```python
eye(size, dtype='float32', name=None)
```
生成一个单位矩阵，示例：
```python
>>> from keras import backend as K
>>> kvar = K.eye(3)
>>> K.eval(kvar)
array([[ 1.,  0.,  0.],
   [ 0.,  1.,  0.],
   [ 0.,  0.,  1.]], dtype=float32)
```


### zeros_like
```python
zeros_like(x, name=None)
```
生成与另一个张量x的shape相同的全0张量，示例：
```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_zeros = K.zeros_like(kvar)
>>> K.eval(kvar_zeros)
array([[ 0.,  0.,  0.],
   [ 0.,  0.,  0.]], dtype=float32)
```

### ones_like
```python
ones_like(x, name=None)
```
生成与另一个张量shape相同的全1张量，示例：
```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_ones = K.ones_like(kvar)
>>> K.eval(kvar_ones)
array([[ 1.,  1.,  1.],
   [ 1.,  1.,  1.]], dtype=float32)
```

### random_uniform_variable
```python
random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None)
```
初始化一个Keras变量，其数值为从一个均匀分布中采样的样本，返回之。

参数：

- shape：张量shape
- low：浮点数，均匀分布之下界
- high：浮点数，均匀分布之上界
- dtype：数据类型
- name：张量名
- seed：随机数种子

示例：
```python
>>> kvar = K.random_uniform_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
>>> K.eval(kvar)
array([[ 0.10940075,  0.10047495,  0.476143  ],
   [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
```

### count_params
```python
count_params(x)
```
返回张量中标量的个数，示例：
```python
>>> kvar = K.zeros((2,3))
>>> K.count_params(kvar)
6
>>> K.eval(kvar)
array([[ 0.,  0.,  0.],
   [ 0.,  0.,  0.]], dtype=float32)
```


###cast
```python
cast(x, dtype)
```
改变张量的数据类型，dtype只能是`float16`, `float32`或`float64`之一，示例：
```python
>>> from keras import backend as K
>>> input = K.placeholder((2, 3), dtype='float32')
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
__It doesn't work in-place as below.__

>>> K.cast(input, dtype='float16')
<tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
__you need to assign it.__

>>> input = K.cast(input, dtype='float16')
>>> input
<tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>```
```

### dot
```python
dot(x, y)
```
求两个张量的乘积。当试图计算两个N阶张量的乘积时，与Theano行为相同，如```(2, 3).(4, 3, 5) = (2, 4, 5))```，示例：
```python
>>> x = K.placeholder(shape=(2, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
```

```python
>>> x = K.placeholder(shape=(32, 28, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
```

Theano-like的行为示例：
```python
>>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
>>> y = K.ones((4, 3, 5))
>>> xy = K.dot(x, y)
>>> K.int_shape(xy)
(2, 4, 5)
```
### batch_dot
```python
batch_dot(x, y, axes=None)
```
按批进行张量乘法，该函数用于计算x和y的点积，其中x和y都是成batch出现的数据。即它的数据shape形如`(batch_size,:)`。batch_dot将产生比输入张量维度低的张量，如果张量的维度被减至1，则通过```expand_dims```保证其维度至少为2
例如，假设```x = [[1, 2],[3,4]]``` ， ```y = [[5, 6],[7, 8]]```，则``` batch_dot(x, y, axes=1) = [[17, 53]] ```，即```x.dot(y.T)```的主对角元素，此过程中我们没有计算过反对角元素的值

参数：

* x,y：阶数大于等于2的张量，在tensorflow下，只支持大于等于3阶的张量
* axes：目标结果的维度，为整数或整数列表，`axes[0]`和`axes[1]`应相同

示例：
假设`x=[[1,2],[3,4]]`，`y=[[5,6],[7,8]]`，则`batch_dot(x, y, axes=1) `为`[[17, 53]]`，恰好为`x.dot(y.T)`的主对角元，整个过程没有计算反对角元的元素。

我们做一下shape的推导，假设x是一个shape为(100,20)的tensor，y是一个shape为(100,30,20)的tensor，假设`axes=(1,2)`，则输出tensor的shape通过循环x.shape和y.shape确定：

- `x.shape[0]`：值为100，加入到输入shape里
- `x.shape[1]`：20，不加入输出shape里，因为该维度的值会被求和(dot_axes[0]=1)
- `y.shape[0]`：值为100，不加入到输出shape里，y的第一维总是被忽略
- `y.shape[1]`：30，加入到输出shape里
- `y.shape[2]`：20，不加到output shape里，y的第二个维度会被求和(dot_axes[1]=2)

- 结果为(100, 30)

```python
>>> x_batch = K.ones(shape=(32, 20, 1))
>>> y_batch = K.ones(shape=(32, 30, 20))
>>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
>>> K.int_shape(xy_batch_dot)
(32, 1, 30)
```
### transpose
```python
transpose(x)
```
张量转置，返回转置后的tensor，示例：
```python
>>> var = K.variable([[1, 2, 3], [4, 5, 6]])
>>> K.eval(var)
array([[ 1.,  2.,  3.],
   [ 4.,  5.,  6.]], dtype=float32)
>>> var_transposed = K.transpose(var)
>>> K.eval(var_transposed)
array([[ 1.,  4.],
   [ 2.,  5.],
   [ 3.,  6.]], dtype=float32)

>>> input = K.placeholder((2, 3))
>>> input
<tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
>>> input_transposed = K.transpose(input)
>>> input_transposed
<tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

```

### gather
```python
gather(reference, indices)
```
在给定的2D张量中检索给定下标的向量

参数：

* reference：2D张量
* indices：整数张量，其元素为要查询的下标

返回值：一个与```reference```数据类型相同的3D张量

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

### var
```python
var(x, axis=None, keepdims=False)
```
在给定轴上计算张量方差

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

### greater
```python
greater(x,y)
```
逐元素判断x>y关系，返回布尔张量

### greater_equal
```python
greater_equal(x,y)
```
逐元素判断x>=y关系，返回布尔张量

### lesser
```python
lesser(x,y)
```
逐元素判断x<y关系，返回布尔张量

### lesser_equal
```python
lesser_equal(x,y)
```
逐元素判断x<=y关系，返回布尔张量

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

### normalize_batch_in_training
```python
normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.0001)
```
对一个batch数据先计算其均值和方差，然后再进行batch_normalization

### batch_normalization
```python
batch_normalization(x, mean, var, beta, gamma, epsilon=0.0001)
```
对一个batch的数据进行batch_normalization，计算公式为：
output = (x-mean)/(sqrt(var)+epsilon)*gamma+beta

###concatenate
```python
concatenate(tensors, axis=-1)
```
在给定轴上将一个列表中的张量串联为一个张量 specified axis

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

参数：

* pattern：代表维度下标的tuple如```(0, 2, 1)```

###resize_images
```python
resize_images(X, height_factor, width_factor, dim_ordering)
```
依据给定的缩放因子，改变一个batch图片的shape，参数中的两个因子都为正整数，图片的排列顺序与维度的模式相关，如‘th’和‘tf’

###resize_volumes
```python
resize_volumes(X, depth_factor, height_factor, width_factor, dim_ordering)
```
依据给定的缩放因子，改变一个5D张量数据的shape，参数中的两个因子都为正整数，图片的排列顺序与维度的模式相关，如‘th’和‘tf’。5D数据的形式是[batch, channels, depth, height, width](th)或[batch, depth, height, width, channels](tf)

###repeat_elements
```python
repeat_elements(x, rep, axis)
```
在给定轴上重复张量元素```rep```次，与```np.repeat```类似。例如，若xshape```(s1, s2, s3) ```并且给定轴为```axis=1`，输出张量的shape为`(s1, s2 * rep, s3)```

###repeat
```python
repeat(x, n)
```
重复2D张量，例如若xshape是```(samples, dim)```且n为2，则输出张量的shape是```(samples, 2, dim)```

###arange
```python
arange(start, stop=None, step=1, dtype='int32')
```
生成1D的整数序列张量，该函数的参数与Theano的arange函数含义相同，如果只有一个参数被提供了，那么它实际上就是`stop`参数的值

为了与tensorflow的默认保持匹配，函数返回张量的默认数据类型是`int32`

### batch_flatten
```python
batch_flatten(x)
```
将一个n阶张量转变为2阶张量，其第一维度保留不变

### expand_dims
```python
expand_dims(x, dim=-1)
```
在下标为```dim```的轴上增加一维

### squeeze
```python
squeeze(x, axis)
```
将下标为```axis```的一维从张量中移除

###temporal_padding
```python
temporal_padding(x, padding=1)
```
向3D张量中间的那个维度的左右两端填充```padding```个0值

###asymmetric_temporal_padding
```python
asymmetric_temporal_padding(x, left_pad=1, right_pad=1)
```
向3D张量中间的那个维度的一端填充```padding```个0值

###spatial_2d_padding
```python
spatial_2d_padding(x, padding=(1, 1), dim_ordering='th')
```
向4D张量第二和第三维度的左右两端填充```padding[0]```和```padding[1]```个0值

###asymmetric_spatial_2d_padding
```python
asymmetric_spatial_2d_padding(x, top_pad=1, bottom_pad=1, left_pad=1, right_pad=1, dim_ordering='th')
```
对4D张量的部分方向进行填充

###spatial_3d_padding
```python
spatial_3d_padding(x, padding=(1, 1, 1), dim_ordering='th')
```
向5D张量深度、高度和宽度三个维度上填充```padding[0]```，```padding[1]```和```padding[2]```个0值


### one-hot
```python
one_hot(indices, nb_classes)
```
输入为n维的整数张量，形如(batch_size, dim1, dim2, ... dim(n-1))，输出为(n+1)维的one-hot编码，形如(batch_size, dim1, dim2, ... dim(n-1), nb_classes)

### reverse
```python
reverse(x, axes)
```
将一个张量在给定轴上反转

###get_value
```python
get_value(x)
```
以Numpy array的形式返回张量的值

###batch_get_value
```python
batch_get_value(x)
```
以Numpy array list的形式返回多个张量的值

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

参数：

* tuples: 列表，其中的元素形如```(tensor, value)```。```value```是要载入的Numpy array数据

### print_tensor
```
print_tensor(x, message='')
```
在求值时打印张量的信息，并返回原张量

###function
```python
function(inputs, outputs, updates=[])
```
实例化一个Keras函数

参数：

* inputs:：列表，其元素为占位符或张量变量
* outputs：输出张量的列表
* updates：列表，其元素是形如```(old_tensor, new_tensor)```的tuple.

###gradients
```python
gradients(loss, variables)
```
返回loss函数关于variables的梯度，variables为张量变量的列表

### stop_gradient
```python
stop_gradient(variables)
```
Returns `variables` but with zero gradient with respect to every other variables.

###rnn
```python
rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```
在张量的时间维上迭代

参数：

* inputs： 形如```(samples, time, ...) ```的时域信号的张量，阶数至少为3
* step_function：每个时间步要执行的函数
	其参数：	
	* input：形如```(samples, ...)```的张量，不含时间维，代表某个时间步时一个batch的样本	
	* states：张量列表
  	其返回值：
	* output：形如```(samples, ...)```的张量
	* new_states：张量列表，与‘states’的长度相同		
* initial_states：形如```(samples, ...)```的张量，包含了```step_function```状态的初始值。
* go_backwards：布尔值，若设为True，则逆向迭代序列
* mask：形如```(samples, time, 1) ```的二值张量，需要屏蔽的数据元素上值为1
* constants：按时间步传递给函数的常数列表
* unroll：当使用TensorFlow时，RNN总是展开的。当使用Theano时，设置该值为```True```将展开递归网络
* input_length：使用TensorFlow时不需要此值，在使用Theano时，如果要展开递归网络，必须指定输入序列

返回值：形如```(last_output, outputs, new_states)```的tuple

* last_output：rnn最后的输出，形如```(samples, ...)```
* outputs：形如```(samples, time, ...) ```的张量，每个在\[s,t\]点的输出对应于样本s在t时间的输出
* new_states: 列表，其元素为形如```(samples, ...)```的张量，代表每个样本的最后一个状态

###switch
```python
switch(condition, then_expression, else_expression)
```
依据给定的条件‘condition’（整数或布尔值）在两个表达式之间切换，注意两个表达式都应该是具有同样shape的符号化张量表达式

参数：

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

参数：

* alpha：负半区斜率
* max_value: 饱和门限

###elu
```python
elu(x, alpha=1.0)
```
指数线性单元

参数：

* x：输入张量
* alpha: 标量

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

###softsign
```python
softsign(x)
```
返回张量的softsign值

###categorical_crossentropy
```python
categorical_crossentropy(output, target, from_logits=False)
```
计算输出张量和目标张量的Categorical crossentropy（类别交叉熵），目标张量与输出张量必须shape相同

###sparse_categorical_crossentropy
```python
sparse_categorical_crossentropy(output, target, from_logits=False)
```
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

参数：

* x：张量
* level：x中设置成0的元素比例
* seed：随机数种子

###l2_normalize
```python
l2_normalize(x, axis)
```
在给定轴上对张量进行L2范数规范化

###in_top_k
```python
in_top_k(predictions, targets, k)
```
判断目标是否在predictions的前k大值位置

参数：

* predictions：预测值张量, shape为(batch_size, classes), 数据类型float32
* targets：真值张量, shape为(batch_size,),数据类型为int32或int64
* k：整数

###conv1d
```python
conv1d(x, kernel, strides=1, border_mode='valid', image_shape=None, filter_shape=None)
```
1D卷积

参数：

* kernel：卷积核张量
* strides：步长，整型
* border_mode：“same”，“valid”之一的字符串

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
* border_mode：“same”或“valid”
* dim_ordering：“tf”或“th”

### conv3d
```python
conv3d(x, kernel, strides=(1, 1, 1), border_mode='valid', dim_ordering='th', volume_shape=None, filter_shape=None)
```
3D卷积

参数：

* x：输入张量
* kernel：卷积核张量
* strides：步长，tuple类型
* border_mode：“same”或“valid”
* dim_ordering：“tf”或“th”

### pool2d
```python
pool2d(x, pool_size, strides=(1, 1), border_mode='valid', dim_ordering='th', pool_mode='max')
```
2D池化

参数：

* pool_size：含有两个整数的tuple，池的大小
* strides：含有两个整数的tuple，步长
* border_mode：“same”，“valid”之一的字符串
* dim_ordering：“tf”和“th”之一，维度排列顺序
* pool_mode: “max”，“avg”之一，池化方式

### pool3d
```python
pool3d(x, pool_size, strides=(1, 1, 1), border_mode='valid', dim_ordering='th', pool_mode='max')
```
3D池化

参数：

* pool_size：含有3个整数的tuple，池的大小
* strides：含有3个整数的tuple，步长
* border_mode：“same”，“valid”之一的字符串
* dim_ordering：“tf”和“th”之一，维度排列顺序
* pool_mode: “max”，“avg”之一，池化方式

### ctc_batch_cost
```python
ctc_batch_cost(y_true, y_pred, input_length, label_length)
```
在batch上运行CTC损失算法

参数：

* y_true：形如(samples，max_tring_length)的张量，包含标签的真值
* y_pred：形如(samples，time_steps，num_categories)的张量，包含预测值或输出的softmax值
* input_length：形如(samples，1)的张量，包含y_pred中每个batch的序列长
* label_length：形如(samples，1)的张量，包含y_true中每个batch的序列长

返回值：形如(samoles，1)的tensor，包含了每个元素的CTC损失

### ctc_decode
```python
ctc_decode(y_pred, input_length, greedy=True, beam_width=None, dict_seq_lens=None, dict_values=None)
```
使用贪婪算法或带约束的字典搜索算法解码softmax的输出

参数：

* y_pred：形如(samples，time_steps，num_categories)的张量，包含预测值或输出的softmax值
* input_length：形如(samples，1)的张量，包含y_pred中每个batch的序列长
* greedy：设置为True使用贪婪算法，速度快
* dict_seq_lens：dic_values列表中各元素的长度
* dict_values：列表的列表，代表字典

返回值：形如(samples，time_steps，num_catgories)的张量，包含了路径可能性（以softmax概率的形式）。注意仍然需要一个用来取出argmax和处理空白标签的函数

### map_fn
```python
map_fn(fn, elems, name=None)
```
元素elems在函数fn上的映射，并返回结果

参数：

* fn：函数
* elems：张量
* name：节点的名字

返回值：返回一个张量，该张量的第一维度等于elems，第二维度取决于fn

### foldl
```python
foldl(fn, elems, initializer=None, name=None)
```
减少elems，用fn从左到右连接它们

参数：

* fn：函数，例如：lambda acc, x: acc + x
* elems：张量
* initializer：初始化的值(elems[0])
* name：节点名

返回值：与initializer的类型和形状一致

### foldr
```python
foldr(fn, elems, initializer=None, name=None)
```
减少elems，用fn从右到左连接它们

参数：

* fn：函数，例如：lambda acc, x: acc + x
* elems：张量	
* initializer：初始化的值（elems[-1]）
* name：节点名

返回值：与initializer的类型和形状一致

### backend
```python
backend()
```
确定当前使用的后端
