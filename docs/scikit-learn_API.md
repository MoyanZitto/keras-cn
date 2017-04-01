# Scikit-Learn接口包装器

我们可以通过包装器将```Sequential```模型（仅有一个输入）作为Scikit-Learn工作流的一部分，相关的包装器定义在```keras.wrappers.scikit_learn.py```中

目前，有两个包装器可用：

```keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)```实现了sklearn的分类器接口


```keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)```实现了sklearn的回归器接口

## 参数

* build_fn：可调用的函数或类对象

* sk_params：模型参数和训练参数

```build_fn```应构造、编译并返回一个Keras模型，该模型将稍后用于训练/测试。```build_fn```的值可能为下列三种之一：

1. 一个函数

2. 一个具有```call```方法的类对象

3. None，代表你的类继承自```KerasClassifier```或```KerasRegressor```，其```call```方法为其父类的```call```方法

```sk_params```以模型参数和训练（超）参数作为参数。合法的模型参数为```build_fn```的参数。注意，‘build_fn’应提供其参数的默认值。所以我们不传递任何值给```sk_params```也可以创建一个分类器/回归器

```sk_params```还接受用于调用```fit```，```predict```，```predict_proba```和```score```方法的参数，如```nb_epoch```，```batch_size```等。这些用于训练或预测的参数按如下顺序选择：

1. 传递给```fit```，```predict```，```predict_proba```和```score```的字典参数

2. 传递个```sk_params```的参数

3. ```keras.models.Sequential```，```fit```，```predict```，```predict_proba```和```score```的默认值

当使用scikit-learn的```grid_search```接口时，合法的可转换参数是你可以传递给```sk_params```的参数，包括训练参数。即，你可以使用```grid_search```来搜索最佳的```batch_size```或```nb_epoch```以及其他模型参数

