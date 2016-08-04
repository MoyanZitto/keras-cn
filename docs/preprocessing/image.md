# 图片预处理

## 图片生成器ImageDataGenerator
```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=True,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    dim_ordering='th')
```
用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。

### 参数

* featurewise_center：布尔值，使输入数据集去中心化（均值为0）

* samplewise_center：布尔值，使输入数据的每个样本均值为0

* featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化

* samplewise_std_normalization：布尔值，将输入的每个样本除以其自身的标准差

* zca_whitening：布尔值，对输入数据施加ZCA白化

* rotation_range：整数，数据提升时图片随机转动的角度

* width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度

* height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度

* shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）

* zoom_range：浮点数或形如```[lower,upper]```的列表，随机缩放的幅度，若为浮点数，则相当于```[lower,upper] = [1 - zoom_range, 1+zoom_range]```

* channel_shift_range：浮点数，随机通道偏移的幅度

* fill_mode：；‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理

* cval：浮点数或整数，当```fill_mode=constant```时，指定要向超出边界的点填充的值

* horizontal_flip：布尔值，进行随机水平翻转

* vertical_flip：布尔值，进行随机竖直翻转

* dim_ordering：‘tf’和‘th’之一，规定数据的维度顺序。‘tf’模式下数据的形状为```samples, width, height, channels```，‘th’下形状为```(samples, channels, width, height)```

***

### 方法

* fit(X, augment=False, rounds=1)：```featurewise_center```，```featurewise_std_normalization```或```zca_whitening```需要此函数。

	* X：numpy array，样本数据
	
	* augment：布尔值，确定是否使用随即提升过的数据
	
	* round：若设```augment=True```，确定要在数据上进行多少轮数据提升，默认值为1
	
* flow(self, X, y, batch_size=32, shuffle=False, seed=None, save_to_dir=None, save_prefix='', save_format='jpeg'):：

	* X：数据
	
	* y：标签
	
	* batch_size：整数，默认32
	
	* shuffle：布尔值，是否随机打乱数据，默认为False
	
	* save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
	
	* save_prefix：字符串，保存提升后图片时使用的前缀
	
	* save_format：‘png’或‘jpeg’之一，指定保存图片的数据格式
	
### 例子

```python
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    samples_per_epoch=len(X_train), nb_epoch=nb_epoch)

# here's a more "manual" example
for e in range(nb_epoch):
    print 'Epoch', e
    batches = 0
    for X_batch, Y_batch in datagen.flow(X_train, Y_train, batch_size=32):
        loss = model.train(X_batch, Y_batch)
        batches += 1
        if batches >= len(X_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```