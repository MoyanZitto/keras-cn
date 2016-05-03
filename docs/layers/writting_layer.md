#编写自己的层

对于简单的定制操作，我们或许可以通过使用```layers.core.Lambda```层来完成。但对于任何具有可训练权重的定制层，你应该自己来实现。

这里是一个Keras层应该具有的框架结构，要定制自己的层，你需要实现下面三个方法

* ```build(input_shape)```：这是定义权重的方法，可训练的权应该在这里被加入列表````self.trainable_weights```中。其他的属性还包括```self.non_trainabe_weights```（列表）和```self.updates```（需要更新的形如（tensor, new_tensor）的tuple的列表）。你可以参考```BatchNormalization```层的实现来学习如何使用上面两个属性。

* ```call(x)```：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心```call```的第一个参数：输入张量

*```get_output_shape_for(input_shape)```：如果你的层修改了输入数据的形状，你应该在这里指定形状变化的方法，这个函数使得Keras可以做自动形状推断

```python
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.random.random((input_dim, output_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] + self.output_dim)
```

现存的Keras层代码可以为你的实现提供良好参考，阅读源代码吧！