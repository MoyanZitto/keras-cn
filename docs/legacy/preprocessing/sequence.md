# 序列预处理

## 填充序列pad_sequences
```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32'，padding='pre', truncating='pre', value=0)
```
将长为```nb_samples```的序列（标量序列）转化为形如```(nb_samples,nb_timesteps)```2D numpy array。如果提供了参数```maxlen```，```nb_timesteps=maxlen```，否则其值为最长序列的长度。其他短于该长度的序列都会在后部填充0以达到该长度。长于`nb_timesteps`的序列将会被截断，以使其匹配目标长度。padding和截断发生的位置分别取决于`padding`和`truncating`.

### 参数

* sequences：浮点数或整数构成的两层嵌套列表

* maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.

* dtype：返回的numpy array的数据类型

* padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补

* truncating：‘pre’或‘post’，确定当需要截断序列时，从起始还是结尾截断

* value：浮点数，此值将在填充时代替默认的填充值0

### 返回值

返回形如```(nb_samples,nb_timesteps)```的2D张量

***

## 跳字skipgrams
```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size, 
    window_size=4, negative_samples=1., shuffle=True, 
    categorical=False, sampling_table=None)
```
skipgrams将一个词向量下标的序列转化为下面的一对tuple：

* 对于正样本，转化为（word，word in the same window）

* 对于负样本，转化为（word，random word from the vocabulary）

【Tips】根据维基百科，n-gram代表在给定序列中产生连续的n项，当序列句子时，每项就是单词，此时n-gram也称为shingles。而skip-gram的推广，skip-gram产生的n项子序列中，各个项在原序列中不连续，而是跳了k个字。例如，对于句子：

“the rain in Spain falls mainly on the plain”

其 2-grams为子序列集合：

the rain，rain in，in Spain，Spain falls，falls mainly，mainly on，on the，the plain

其 1-skip-2-grams为子序列集合：

the in, rain Spain, in falls, Spain mainly, falls on, mainly the, on plain.

更多详情请参考[<font color='FF0000'>Efficient Estimation of Word Representations in Vector Space</font>](http://arxiv.org/pdf/1301.3781v3.pdf)【@BigMoyan】

### 参数

* sequence：下标的列表，如果使用sampling_tabel，则某个词的下标应该为它在数据库中的顺序。（从1开始）

* vocabulary_size：整数，字典大小

* window_size：整数，正样本对之间的最大距离

* negative_samples：大于0的浮点数，等于0代表没有负样本，等于1代表负样本与正样本数目相同，以此类推（即负样本的数目是正样本的```negative_samples```倍）

* shuffle：布尔值，确定是否随机打乱样本

* categorical：布尔值，确定是否要使得返回的标签具有确定类别

* sampling_table：形如```(vocabulary_size,)```的numpy array，其中```sampling_table[i]```代表没有负样本或随机负样本。等于1为与正样本的数目相同
采样到该下标为i的单词的概率（假定该单词是数据库中第i常见的单词）

### 输出

函数的输出是一个```(couples,labels)```的元组，其中：

* ```couples```是一个长为2的整数列表：```[word_index,other_word_index]```

* ```labels```是一个仅由0和1构成的列表，1代表```other_word_index```在```word_index```的窗口，0代表```other_word_index```是词典里的随机单词。

* 如果设置```categorical```为```True```，则标签将以one-hot的方式给出，即1变为\[0,1\]，0变为\[1,0\]

***

## 获取采样表make_sampling_table
```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-5)
```
该函数用以产生```skipgrams```中所需要的参数```sampling_table```。这是一个长为```size```的向量，```sampling_table[i]```代表采样到数据集中第i常见的词的概率（为平衡期起见，对于越经常出现的词，要以越低的概率采到它）

### 参数

* size：词典的大小

* sampling_factor：此值越低，则代表采样时更缓慢的概率衰减（即常用的词会被以更低的概率被采到），如果设置为1，则代表不进行下采样，即所有样本被采样到的概率都是1。


