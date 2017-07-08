*这里需要说明一下，笔者**不建议在Windows环境下进行深度学习的研究**，一方面是因为Windows所对应的框架搭建的依赖过多，社区设定不完全；另一方面，Linux系统下对显卡支持、内存释放以及存储空间调整等硬件功能支持较好。如果您对Linux环境感到陌生，并且大多数开发环境在Windows下更方便操作的话，希望这篇文章对您会有帮助。*


**由于Keras默认以Tensorflow为后端，且Theano后端更新缓慢，本文默认采用Tensorflow1.0作为Keras后端，Theano版安装方式请访问[www.scp-173.top**](http://www.scp-173.top)

---
# 关于计算机的硬件配置说明
## **推荐配置**
如果您是高校学生或者高级研究人员，并且实验室或者个人资金充沛，建议您采用如下配置：

 - 主板：X299型号或Z270型号
 - CPU:  i7-6950X或i7-7700K 及其以上高级型号
 - 内存：品牌内存，总容量32G以上，根据主板组成4通道或8通道
 - SSD： 品牌固态硬盘，容量256G以上
 - <font color=#FF0000>显卡：NVIDIA GTX TITAN(XP) NVIDIA GTX 1080ti、NVIDIA GTX TITAN、NVIDIA GTX 1080、NVIDIA GTX 1070、NVIDIA GTX 1060 (顺序为优先建议，并且建议同一显卡，可以根据主板插槽数量购买多块，例如X299型号主板最多可以采用×4的显卡)</font>
 - 电源：由主机机容量的确定，一般有显卡总容量后再加200W即可
## **最低配置**
如果您是仅仅用于自学或代码调试，亦或是条件所限仅采用自己现有的设备进行开发，那么您的电脑至少满足以下几点：

 - CPU：Intel第三代i5和i7以上系列产品或同性能AMD公司产品
 - 内存：总容量4G以上

## <font color=#FF0000>CPU说明</font>
 - 大多数CPU目前支持多核多线程，那么如果您采用CPU加速，就可以使用多线程运算。这方面的优势对于服务器CPU志强系列尤为关键
## <font color=#FF0000>显卡说明</font>
 - 如果您的显卡是非NVIDIA公司的产品或是NVIDIA GTX系列中型号的第一个数字低于6或NVIDIA的GT系列，都不建议您采用此类显卡进行加速计算，例如`NVIDIA GT 910`、`NVIDIA GTX 460` 等等。
 - 如果您的显卡为笔记本上的GTX移动显卡（型号后面带有标识M），那么请您慎重使用显卡加速，因为移动版GPU容易发生过热烧毁现象。
 - 如果您的显卡，显示的是诸如 `HD5000`,`ATI 5650` 等类型的显卡，那么您只能使用CPU加速
 - 如果您的显卡芯片为Pascal架构（`NVIDIA GTX 1080`,`NVIDIA GTX 1070`等），您只能在之后的配置中选择`CUDA 8.0`
 ---

# 基本开发环境搭建
## 1. Microsoft Windows 版本
关于Windows的版本选择，本人强烈建议对于部分高性能的新机器采用`Windows 10`作为基础环境，部分老旧笔记本或低性能机器采用`Windows 7`即可，本文环境将以`Windows 10`作为开发环境进行描述。对于Windows 10的发行版本选择，笔者建议采用`Windows_10_enterprise_2016_ltsb_x64`作为基础环境。

这里推荐到[<font color=#FF0000>MSDN我告诉你</font>](http://msdn.itellyou.cn/)下载，也感谢作者国内优秀作者[雪龙狼前辈](http://weibo.com/207156000?is_hot=1)所做出的贡献与牺牲。

![](../images/keras_windows_1.png)

直接贴出热链，复制粘贴迅雷下载：

    ed2k://|file|cn_windows_10_enterprise_2016_ltsb_x64_dvd_9060409.iso|3821895680|FF17FF2D5919E3A560151BBC11C399D1|/


## 2. 编译环境Microsoft Visual Studio 2015 Update 3
*<font color=#FF0000>(安装CPU版本非必须安装)</font>*

CUDA编译器为Microsoft Visual Studio，版本从2010-2015，`cuda8.0`仅支持2015版本，暂不支持VS2017，本文采用`Visual Studio 2015 Update 3`。
同样直接贴出迅雷热链：

    ed2k://|file|cn_visual_studio_professional_2015_with_update_3_x86_x64_dvd_8923256.iso|7745202176|DD35D3D169D553224BE5FB44E074ED5E|/
 ![MSDN](../images/keras_windows_2.png)

## 3. Python环境
python环境建设推荐使用科学计算集成python发行版**Anaconda**，Anaconda是Python众多发行版中非常适用于科学计算的版本，里面已经集成了很多优秀的科学计算Python库。
建议安装`Anconda3 4.2.0`版本，目前新出的python3.6存在部分不兼容问题，所以建议安装历史版本4.2.0
**注意：windows版本下的tensorflow暂时不支持python2.7**

下载地址： [<font color=#FF0000>Anaconda</font>](https://repo.continuum.io/archive/index.html)


## 4. CUDA
*<font color=#FF0000>(安装CPU版本非必须安装)</font>*
CUDA Toolkit是NVIDIA公司面向GPU编程提供的基础工具包，也是驱动显卡计算的核心技术工具。
直接安装CUDA8.0即可
下载地址：[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
![](../images/keras_windows_3.png)
在下载之后，按照步骤安装，**不建议新手修改安装目录**，同上，环境不需要配置，安装程序会自动配置好。

## 6. 加速库CuDNN
从官网下载需要注册 Nvidia 开发者账号，网盘搜索一般也能找到。
Windows目前最新版v6.0，但是keras尚未支持此版本，请下载v5.1版本，即 cudnn-8.0-win-x64-v5.1.zip。
下载解压出来是名为cuda的文件夹，里面有bin、include、lib，将三个文件夹复制到安装CUDA的地方覆盖对应文件夹，默认文件夹在：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\8.0`

---

# Keras 框架搭建
## 安装

在CMD命令行或者Powershell中输入：
``` powershell
# GPU 版本
>>> pip install --upgrade tensorflow-gpu

# CPU 版本
>>> pip install --upgrade tensorflow

# Keras 安装
>>> pip install keras -U --pre
```

之后可以验证keras是否安装成功,在命令行中输入Python命令进入Python变成命令行环境：
```python
>>> import keras

Using Tensorflow backend.
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:135] successfully opened CUDA library cublas64_80.dll locally
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:135] successfully opened CUDA library cudnn64_5.dll locally
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:135] successfully opened CUDA library cufft64_80.dll locally
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:135] successfully opened CUDA library nvcuda.dll locally
I c:\tf_jenkins\home\workspace\release-win\device\gpu\os\windows\tensorflow\stream_executor\dso_loader.cc:135] successfully opened CUDA library curand64_80.dll locally

>>>
```
没有报错，那么Keras就已经**成功安装**了


 - Keras中mnist数据集测试
 下载Keras开发包
```
>>> conda install git
>>> git clone https://github.com/fchollet/keras.git
>>> cd keras/examples/
>>> python mnist_mlp.py
```
程序无错进行，至此，keras安装完成。

[<font color='#FF0000'>Keras中文文档地址</font>](http://keras-cn.readthedocs.io/)

## 声明与联系方式 ##

由于作者水平和研究方向所限，无法对所有模块都非常精通，因此文档中不可避免的会出现各种错误、疏漏和不足之处。如果您在使用过程中有任何意见、建议和疑问，欢迎发送邮件到scp173.cool@gmail.com与中文文档作者取得联系.

**本教程不得用于任何形式的商业用途，如果需要转载请与作者或中文文档作者联系，如果发现未经允许复制转载，将保留追求其法律责任的权利。**

作者：[SCP-173](https://github.com/KaiwenXiao)
E-mail ：scp173.cool@gmail.com
**如果您需要及时得到指导帮助，可以加微信：SCP173-cool，酌情打赏即可**
![微信](../images/scp_173.png)
