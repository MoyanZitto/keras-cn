# 图片预处理

## 图片生成器ImageDataGenerator
```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
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
    rescale=None,
    dim_ordering=K.image_dim_ordering())
```
用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。

### 参数

* featurewise_center：布尔值，使输入数据集去中心化（均值为0）, 按feature执行

* samplewise_center：布尔值，使输入数据的每个样本均值为0

* featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行

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

* rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)

* dim_ordering：‘tf’和‘th’之一，规定数据的维度顺序。‘tf’模式下数据的形状为```samples, height, width, channels```，‘th’下形状为```(samples, channels, height, width).```该参数的默认值是Keras配置文件```~/.keras/keras.json```的```image_dim_ordering```值,如果你从未设置过的话,就是'tf'

***

### 方法

* fit(X, augment=False, rounds=1)：计算依赖于数据的变换所需要的统计信息(均值方差等),只有使用```featurewise_center```，```featurewise_std_normalization```或```zca_whitening```时需要此函数。

	* X：numpy array，样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
	
	* augment：布尔值，确定是否使用随即提升过的数据
	
	* round：若设```augment=True```，确定要在数据上进行多少轮数据提升，默认值为1
	
	* seed: 整数,随机数种子
	
* flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='jpeg')：接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据

	* X：样本数据，秩应为4.在黑白图像的情况下channel轴的值为1，在彩色图像情况下值为3
	
	* y：标签
	
	* batch_size：整数，默认32
	
	* shuffle：布尔值，是否随机打乱数据，默认为True
	
	* save_to_dir：None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
	
	* save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了```save_to_dir```时生效
	
	* save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
	
	* yields:形如(x,y)的tuple,x是代表图像数据的numpy数组.y是代表标签的numpy数组.该迭代器无限循环.

	* seed: 整数,随机数种子

* flow_from_directory(directory): 以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据

	* directory: 目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG和BNP的图片都会被生成器使用.详情请查看[<font color='#FF0000'>此脚本</font>](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
    * target_size: 整数tuple,默认为(256, 256). 图像将被resize成该尺寸
    * color_mode: 颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.
    * classes: 可选参数,为子文件夹的列表,如['dogs','cats']默认为None. 若未提供,则该类别列表将自动推断(类别的顺序将按照字母表顺序映射到标签值)
    * class_mode: "categorical", "binary", "sparse"或None之一. 默认为"categorical. 该参数决定了返回的标签数组的形式, "categorical"会返回2D的one-hot编码标签,"binary"返回1D的二值标签."sparse"返回1D的整数标签,如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 这种情况在使用```model.predict_generator()```和```model.evaluate_generator()```等函数时会用到.
    * batch_size: batch数据的大小,默认32
    * shuffle: 是否打乱数据,默认为True
    * seed: 可选参数,打乱数据和进行变换时的随机数种子
	* save_to_dir: None或字符串，该参数能让你将提升后的图片保存起来，用以可视化
	* save_prefix：字符串，保存提升后图片时使用的前缀, 仅当设置了```save_to_dir```时生效
	* save_format："png"或"jpeg"之一，指定保存图片的数据格式,默认"jpeg"
    * flollow_links: 是否访问子文件夹中的软链接

	
### 例子

使用```.flow()```的例子
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

使用```.flow_from_directory(directory)```的例子
```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=50,
        validation_data=validation_generator,
        nb_val_samples=800)
```

同时变换图像和mask

```python
# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=50)
```
