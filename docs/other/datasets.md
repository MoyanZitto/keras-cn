# 常用数据库

## CIFAR10 小图片分类数据集

该数据库具有50,000个32*32的彩色图片作为训练集，10,000个图片作为测试集。图片一共有10个类别。

### 使用方法
```python
from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

### 返回值：

两个Tuple

```X_train```和```X_test```是形如（nb_samples, 3, 32, 32）的RGB三通道图像数据，数据类型是无符号8位整形（uint8）

```Y_train```和 ```Y_test```是形如（nb_samples,）标签数据，标签的范围是0~9

***

## CIFAR100 小图片分类数据库

该数据库具有50,000个32*32的彩色图片作为训练集，10,000个图片作为测试集。图片一共有100个类别，每个类别有600张图片。这100个类别又分为20个大类。

### 使用方法
```python
from keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
```

### 参数

* label_mode：为‘fine’或‘coarse’之一，控制标签的精细度，‘fine’获得的标签是100个小类的标签，‘coarse’获得的标签是大类的标签

### 返回值

两个Tuple,```(X_train, y_train), (X_test, y_test)```，其中

* X_train和X_test：是形如（nb_samples, 3, 32, 32）的RGB三通道图像数据，数据类型是无符号8位整形（uint8）

* y_train和y_test：是形如（nb_samples,）标签数据，标签的范围是0~9

***

## IMDB影评倾向分类

本数据库含有来自IMDB的25,000条影评，被标记为正面/负面两种评价。影评已被预处理为词下标构成的[<font color='#FF0000'>序列</font>](../preprocessing/sequence)。方便起见，单词的下标基于它在数据集中出现的频率标定，例如整数3所编码的词为数据集中第3常出现的词。这样的组织方法使得用户可以快速完成诸如“只考虑最常出现的10,000个词，但不考虑最常出现的20个词”这样的操作

按照惯例，0不代表任何特定的词，而用来编码任何未知单词

### 使用方法
```python
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      nb_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      test_split=0.1)
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
```

### 参数

* path：如果你在本机上已有此数据集（位于```'~/.keras/datasets/'+path```），则载入。否则数据将下载到该目录下

* nb_words：整数或None，要考虑的最常见的单词数，序列中任何出现频率更低的单词将会被编码为`oov_char`的值。

* skip_top：整数，忽略最常出现的若干单词，这些单词将会被编码为`oov_char`的值

* maxlen：整数，最大序列长度，任何长度大于此值的序列将被截断

* seed：整数，用于数据重排的随机数种子

* start_char：字符，序列的起始将以该字符标记，默认为1因为0通常用作padding

* oov_char：整数，因```nb_words```或```skip_top```限制而cut掉的单词将被该字符代替

* index_from：整数，真实的单词（而不是类似于```start_char```的特殊占位符）将从这个下标开始

### 返回值

两个Tuple,```(X_train, y_train), (X_test, y_test)```，其中

* X_train和X_test：序列的列表，每个序列都是词下标的列表。如果指定了```nb_words```，则序列中可能的最大下标为```nb_words-1```。如果指定了```maxlen```，则序列的最大可能长度为```maxlen```

* y_train和y_test：为序列的标签，是一个二值list

***

## 路透社新闻主题分类

本数据库包含来自路透社的11,228条新闻，分为了46个主题。与IMDB库一样，每条新闻被编码为一个词下标的序列。

### 使用方法
```python
from keras.datasets import reuters


(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         nb_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```

参数的含义与IMDB同名参数相同，唯一多的参数是：
```test_split```，用于指定从原数据中分割出作为测试集的比例。该数据库支持获取用于编码序列的词下标：
```python
word_index = reuters.get_word_index(path="reuters_word_index.json")
```
上面代码的返回值是一个以单词为关键字，以其下标为值的字典。例如，```word_index['giraffe']```的值可能为```1234```

### 参数

* path：如果你在本机上已有此数据集（位于```'~/.keras/datasets/'+path```），则载入。否则数据将下载到该目录下

***

## MNIST手写数字识别

本数据库有60,000个用于训练的28*28的灰度手写数字图片，10,000个测试图片

### 使用方法
```python
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
### 参数

* path：如果你在本机上已有此数据集（位于```'~/.keras/datasets/'+path```），则载入。否则数据将下载到该目录下

### 返回值

两个Tuple,```(X_train, y_train), (X_test, y_test)```，其中

* X_train和X_test：是形如（nb_samples, 28, 28）的灰度图片数据，数据类型是无符号8位整形（uint8）

* y_train和y_test：是形如（nb_samples,）标签数据，标签的范围是0~9

数据库将会被下载到```'~/.keras/datasets/'+path```

***

## Fashion-MNIST数据集

本数据集包含60,000个28x28灰度图像，共10个时尚分类作为训练集。测试集包含10,000张图片。该数据集可作为MNIST数据集的进化版本，10个类别标签分别是： 

|类别|描述|
|:-:|:-:|
|0|T恤/上衣|
|1|裤子|
|2|套头衫|
|3|连衣裙|
|4|大衣|
|5|凉鞋|
|6|衬衫|
|7|帆布鞋|
|8|包|
|9|短靴|

### 使用方法
```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```
### 参数

* path：如果你在本机上已有此数据集（位于```'~/.keras/datasets/'+path```），则载入。否则数据将下载到该目录下

### 返回值

两个Tuple,```(X_train, y_train), (X_test, y_test)```，其中

* X_train和X_test：是形如（nb_samples, 28, 28）的灰度图片数据，数据类型是无符号8位整形（uint8）

* y_train和y_test：是形如（nb_samples,）标签数据，标签的范围是0~9

数据库将会被下载到```'~/.keras/datasets/'+path```

***

## Boston房屋价格回归数据集

本数据集由StatLib库取得，由CMU维护。每个样本都是1970s晚期波士顿郊区的不同位置，每条数据含有13个属性，目标值是该位置房子的房价中位数（千dollar）。


### 使用方法
```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

### 参数

* path：数据存放位置，默认```'~/.keras/datasets/'+path```

* seed：随机数种子

* test_split：分割测试集的比例

### 返回值

两个Tuple,```(X_train, y_train), (X_test, y_test)```

