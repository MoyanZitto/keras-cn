# 数据工具

## get_file

```python
get_file(fname, origin, untar=False, md5_hash=None, cache_subdir='datasets')
```

从给定的URL中下载文件, 可以传递MD5值用于数据校验(下载后或已经缓存的数据均可)

### 参数

* fname: 文件名

* origin: 文件的URL地址

* untar: 布尔值,是否要进行解压

* md5_hash: MD5哈希值,用于数据校验

* cache_subdir: 用于缓存数据的文件夹

### 返回值

下载后的文件地址
