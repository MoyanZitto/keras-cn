# I/O工具

## HDF5矩阵

```python
keras.utils.io_utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

这是一个使用HDF5数据集代替Numpy数组的方法。

### 例子
```python
X_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(X_data)
```

提供start和end参数可以使用数据集的切片。

可选的，可以给出归一化函数（或lambda表达式）。 这将在每个检索的数据集的切片上调用。

### 参数

* datapath：字符串，HDF5文件的路径

* dataset：字符串，在datapath中指定的文件中的HDF5数据集的名称

* start：整数，指定数据集的所需切片的开始

* end：整数，指定数据集的所需切片的结尾

* normalizer：数据集在被检索时的调用函数
