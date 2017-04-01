#编写自己的层


对于简单的定制操作，我们或许可以通过使用```layers.core.Lambda```层来完成。但对于任何具有可训练权重的定制层，你应该自己来实现。

这里是一个Keras层应该具有的框架结构(1.1.3以后的版本，如果你的版本更旧请升级)，要定制自己的层，你需要实现下面三个方法

* ```build(input_shape)```：这是定义权重的方法，可训练的权应该在这里被加入列表````self.trainable_weights```中。其他的属性还包括```self.non_trainabe_weights```（列表）和```self.updates```（需要更新的形如（tensor, new_tensor）的tuple的列表）。你可以参考```BatchNormalization```层的实现来学习如何使用上面两个属性。这个方法必须设置```self.built = True```，可通过调用```super([layer],self).build()```实现

* ```call(x)```：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心```call```的第一个参数：输入张量

* ```get_output_shape_for(input_shape)```：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断

```python
from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
                                initializer='random_uniform',
                                trainable=True)
        super(MyLayer, self).build()  # be sure you call this somewhere! 

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0] + self.output_dim)
```
<a name='adjust'>
<font color='#404040'>
## 调整旧版Keras编写的层以适应Keras1.0
</font>
</a>

以下内容是你在将旧版Keras实现的层调整为新版Keras应注意的内容，这些内容对你在Keras1.0中编写自己的层也有所帮助。

* 你的Layer应该继承自```keras.engine.topology.Layer```，而不是之前的```keras.layers.core.Layer```。另外，```MaskedLayer```已经被移除。

* ```build```方法现在接受```input_shape```参数，而不是像以前一样通过```self.input_shape```来获得该值，所以请把```build(self)```转为```build(self, input_shape)```

* 请正确将```output_shape```属性转换为方法```get_output_shape_for(self, train=False)```，并删去原来的```output_shape```

* 新层的计算逻辑现在应实现在```call```方法中，而不是之前的```get_output```。注意不要改动```__call__```方法。将```get_output(self,train=False)```转换为```call(self,x,mask=None)```后请删除原来的```get_output```方法。

* Keras1.0不再使用布尔值```train```来控制训练状态和测试状态，如果你的层在测试和训练两种情形下表现不同，请在```call```中使用指定状态的函数。如，```x=K.in_train_phase(train_x, test_y)```。例如，在Dropout的```call```方法中你可以看到：

```python
return K.in_train_phase(K.dropout(x, level=self.p), x)
```

* ```get_config```返回的配置信息可能会包括类名，请从该函数中将其去掉。如果你的层在实例化时需要更多信息（即使将```config```作为kwargs传入也不能提供足够信息），请重新实现```from_config```。请参考```Lambda```或```Merge```层看看复杂的```from_config```是如何实现的。

* 如果你在使用Masking，请实现```compute_mas(input_tensor, input_mask)```，该函数将返回```output_mask```。请确保在```__init__()```中设置```self.supports_masking = True```

* 如果你希望Keras在你编写的层与Keras内置层相连时进行输入兼容性检查，请在```__init__```设置```self.input_specs```或实现```input_specs()```并包装为属性（@property）。该属性应为```engine.InputSpec```的对象列表。在你希望在```call```中获取输入shape时，该属性也比较有用。

* 下面的方法和属性是内置的，请不要覆盖它们

	* ```__call__```
	
	* ```add_input```
	
	* ```assert_input_compatibility```
	
	* ```set_input```
	
	* ```input```
	
	* ```output```
	
	* ```input_shape```

	* ```output_shape```
	
	* ```input_mask```
	
	* ```output_mask```
	
	* ```get_input_at```
	
	* ```get_output_at```
	
	* ```get_input_shape_at```

	* ```get_output_shape_at```
	
	* ```get_input_mask_at```
	
	* ```get_output_mask_at```
	

现存的Keras层代码可以为你的实现提供良好参考，阅读源代码吧！
