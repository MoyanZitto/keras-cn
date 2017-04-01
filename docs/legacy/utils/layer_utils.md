# Keras层工具

## layer_from_config

```python
layer_from_config(config, custom_objects={})
```
从配置生成Keras层对象

### 参数

* config:形如{'class_name':str, 'config':dict}的字典

* custom_objects: 字典,用以将定制的非Keras对象之类名/函数名映射为类/函数对象

### 返回值

层对象,包含Model,Sequential和其他Layer
