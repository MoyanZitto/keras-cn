# 关于Keras的“层”（Layer）

所有的Keras层对象都有如下方法：

* ```layer.get_weights()```：返回层的权重（numpy array）

* ```layer.set_weights(weights)```：从numpy array中将权重加载到该层中，要求numpy array的形状与* ```layer.get_weights()```的形状相同

* ```layer.get_config()```：返回当前层配置信息的字典，层也可以借由配置信息重构:
```python
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)
```

或者：

```python
from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})
```

如果层仅有一个计算节点（即该层不是共享层），则可以通过下列方法获得输入张量、输出张量、输入数据的形状和输出数据的形状：

* ```layer.input```

* ```layer.output```

* ```layer.input_shape```

* ```layer.output_shape```

如果该层有多个计算节点（参考[层计算节点和共享层](../getting_started/functional_API/#node)）。可以使用下面的方法

* ```layer.get_input_at(node_index)```

* ```layer.get_output_at(node_index)```

* ```layer.get_input_shape_at(node_index)```

* ```layer.get_output_shape_at(node_index)```