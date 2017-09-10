#编写自己的层


对于简单的定制操作，我们或许可以通过使用```layers.core.Lambda```层来完成。但对于任何具有可训练权重的定制层，你应该自己来实现。

这里是一个Keras2的层应该具有的框架结构(如果你的版本更旧请升级)，要定制自己的层，你需要实现下面三个方法

* ```build(input_shape)```：这是定义权重的方法，可训练的权应该在这里被加入列表````self.trainable_weights```中。其他的属性还包括```self.non_trainabe_weights```（列表）和```self.updates```（需要更新的形如（tensor, new_tensor）的tuple的列表）。你可以参考```BatchNormalization```层的实现来学习如何使用上面两个属性。这个方法必须设置```self.built = True```，可通过调用```super([layer],self).build()```实现

* ```call(x)```：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心```call```的第一个参数：输入张量

* ```compute_output_shape(input_shape)```：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断

```python
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

现存的Keras层代码可以为你的实现提供良好参考，阅读源代码吧！
