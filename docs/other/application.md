# Application应用

Kera的应用模块Application提供了带有预训练权重的Keras模型，这些模型可以用来进行预测、特征提取和finetune

模型的预训练权重将下载到```~/.keras/models/```并在载入模型时自动载入

## 可用的模型

应用于图像分类的预训练权重训练自ImageNet：

* [VGG16](#vgg16)
* [VGG19](#vgg19)
* [ResNet50](#resnet50)
* [InceptionV3](#inceptionv3)

所有的这些模型都兼容Theano和Tensorflow，并会自动基于```~/.keras/keras.json```的Keras的图像维度进行自动设置。例如，如果你设置```image_dim_ordering=tf```，则加载的模型将按照TensorFlow的维度顺序来构造，即“Width-Height-Depth”的顺序

音频文件自动标注模型（梅尔谱图作为输入）

* [MusicTaggerCRNN](#musictaggercrnn)

***

## 图片分类模型的示例

### 利用ResNet50网络进行ImageNet分类
```python
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
# print: [[u'n02504458', u'African_elephant']]
```

### 利用VGG16提取特征
```python
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### 从VGG19的任意中间层中抽取特征
```python
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

### 利用新数据集finetune InceptionV3
```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)
```

### 在定制的输入tensor上构建InceptionV3
```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_dim_ordering() == 'tf'

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```

***

模型文档

* [VGG16](#vgg16)
* [VGG19](#vgg19)
* [ResNet50](#resnet50)
* [InceptionV3](#inceptionv3)
* [MusicTaggerCRNN](#musictaggercrnn)

***

<a name='vgg16'>
<font color='#404040'>
## VGG16模型
</font>
</a>
```python
keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None)
```

### 参数
* include_top：是否保留顶层的3个全连接网络
* weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
* input_tensor：可填入Keras tensor作为模型的图像输出tensor

### 返回值

Keras 模型对象

### 参考文献

* [<font color='#FF0000'>Very Deep Convolutional Networks for Large-Scale Image Recognition</font>](https://arxiv.org/abs/1409.1556)：如果在研究中使用了VGG，请引用该文

### License
预训练权重由[<font color='#FF0000'>牛津VGG组</font>](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)发布的预训练权重移植而来，基于[<font color='#FF0000'>Creative Commons Attribution License</font>](https://creativecommons.org/licenses/by/4.0/)

***

<a name='vgg19'>
<font color='#404040'>
## VGG19模型
</font>
</a>
```python
keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None)
```

### 参数
* include_top：是否保留顶层的3个全连接网络
* weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
* input_tensor：可填入Keras tensor作为模型的图像输出tensor

### 返回值

Keras 模型对象

### 参考文献

* [<font color='#FF0000'>Very Deep Convolutional Networks for Large-Scale Image Recognition</font>](https://arxiv.org/abs/1409.1556)：如果在研究中使用了VGG，请引用该文

### License
预训练权重由[<font color='#FF0000'>牛津VGG组</font>](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)发布的预训练权重移植而来，基于[<font color='#FF0000'>Creative Commons Attribution License</font>](https://creativecommons.org/licenses/by/4.0/)

***

<a name='resnet50'>
<font color='#404040'>
## ResNet50模型
</font>
</a>
```python
keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None)
```

### 参数
* include_top：是否保留顶层的3个全连接网络
* weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
* input_tensor：可填入Keras tensor作为模型的图像输出tensor

### 返回值

Keras 模型对象

### 参考文献

* [<font color='#FF0000'>Deep Residual Learning for Image Recognition</font>](https://arxiv.org/abs/1512.03385)：如果在研究中使用了ResNet50，请引用该文

### License
预训练权重由[<font color='#FF0000'>Kaiming He</font>](https://github.com/KaimingHe/deep-residual-networks)发布的预训练权重移植而来，基于[<font color='#FF0000'>MIT License</font>](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE)

***

<a name='inceptionv3'>
<font color='#404040'>
## InceptionV3模型
</font>
</a>
```python
keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None)
```

### 参数
* include_top：是否保留顶层的3个全连接网络
* weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
* input_tensor：可填入Keras tensor作为模型的图像输出tensor

### 返回值

Keras 模型对象

### 参考文献

* [<font color='#FF0000'>Rethinking the Inception Architecture for Computer Vision</font>](http://arxiv.org/abs/1512.00567)：如果在研究中使用了InceptionV3，请引用该文

### License
预训练权重由我们自己训练而来，基于[<font color='#FF0000'>MIT License</font>](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE)

***

<a name='musictaggercrnn'>
<font color='#404040'>
## MusicTaggerCRNN模型
</font>
</a>
```python
keras.applications.music_tagger_crnn.MusicTaggerCRNN(weights='msd', input_tensor=None, include_top=True)
```

卷积循环模型将音乐曲目的MelSpectrogram的矢量化表示作为输入，并且能够输出曲目的音乐流派。 您可以使用```keras.applications.music_tagger_crnn.preprocess_input```将声音文件转换为矢量化频谱图。 这需要安装[<font color='#FF0000'>Librosa</font>](http://librosa.github.io/librosa/)库。 请参阅[<font color='#FF0000'>示例</font>](#music-tagging-and-feature-extraction-with-musictaggercrnn)。

### 参数
* weights：None代表随机初始化，即不加载预训练权重。'msd'代表加载预训练权重[<font color='#FF0000'>Million Song Dataset</font>](http://labrosa.ee.columbia.edu/millionsong/)
* input_tensor：可填入Keras tensor作为模型的图像输出tensor
* include_top：是否保留顶层的1个全连接网络，如为否输出32维特征

### 返回值

Keras 模型对象

### 参考文献

* [<font color='#FF0000'>Convolutional Recurrent Neural Networks for Music Classification</font>](https://arxiv.org/abs/1609.04243)：如果在研究中使用了MusicTaggerCRNN，请引用该文

### License
预训练权重由[<font color='#FF0000'>Keunwoo Choi</font>](https://github.com/keunwoochoi/music-auto_tagging-keras)发布的预训练权重移植而来，基于[<font color='#FF0000'>MIT License</font>](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE)

<a name='music-tagging-and-feature-extraction-with-musictaggercrnn'>
### 例子：音乐标记和音频特征提取
</a>
```python
from keras.applications.music_tagger_crnn import MusicTaggerCRNN
from keras.applications.music_tagger_crnn import preprocess_input, decode_predictions
import numpy as np

# 1. Tagging
model = MusicTaggerCRNN(weights='msd')

audio_path = 'audio_file.mp3'
melgram = preprocess_input(audio_path)
melgrams = np.expand_dims(melgram, axis=0)

preds = model.predict(melgrams)
print('Predicted:')
print(decode_predictions(preds))
# print: ('Predicted:', [[('rock', 0.097071797), ('pop', 0.042456303), ('alternative', 0.032439161), ('indie', 0.024491295), ('female vocalists', 0.016455274)]])

#. 2. Feature extraction
model = MusicTaggerCRNN(weights='msd', include_top=False)

audio_path = 'audio_file.mp3'
melgram = preprocess_input(audio_path)
melgrams = np.expand_dims(melgram, axis=0)

feats = model.predict(melgrams)
print('Features:')
print(feats[0, :10])
# print: ('Features:', [-0.19160545 0.94259131 -0.9991011 0.47644514 -0.19089699 0.99033844 0.1103896 -0.00340496 0.14823607 0.59856361])
```

