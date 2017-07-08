**本教程不得用于任何形式的商业用途，如果需要转载请与作者SCP-173联系，如果发现未经允许复制转载，将保留追求其法律责任的权利。**



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
## 1. Linux 发行版
linux有很多发行版，本文强烈建议读者采用新版的`Ubuntu 16.04 LTS`
一方面，对于大多数新手来说Ubuntu具有很好的图形界面，与乐观的开源社区；另一方面，Ubuntu是Nvidia官方以及绝大多数深度学习框架默认开发环境。
个人不建议使用Ubuntu其他版本，由于GCC编译器版本不同，会导致很多依赖无法有效安装。
Ubuntu 16.04 LTS<font color=#FF0000>下载地址</font>：http://www.ubuntu.org.cn/download/desktop
![](../images/keras_ubuntu_1.png)
通过U盘安装好后，进行初始化环境设置。
## 2. Ubuntu初始环境设置

 - 安装开发包
打开`终端`输入：
```bash
# 系统升级
>>> sudo apt update
>>> sudo apt upgrade
# 安装python基础开发包
>>> sudo apt install -y python-dev python-pip python-nose gcc g++ git gfortran vim
```

 - 安装运算加速库
打开`终端`输入：
```
>>> sudo apt install -y libopenblas-dev liblapack-dev libatlas-base-dev
```

## 3. CUDA开发环境的搭建(CPU加速跳过)
***如果您的仅仅采用cpu加速，可跳过此步骤***
 - 下载CUDA8.0

下载地址：https://developer.nvidia.com/cuda-downloads
![](../images/keras_ubuntu_2.png)

之后打开`终端`输入：

```
>>> sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
>>> sudo apt update
>>> sudo apt -y install cuda
```
自动配置成功就好。

 - 将CUDA路径添加至环境变量
在`终端`输入：
```
>>> sudo gedit /etc/profile
```
在`profile`文件中添加：
```bash
export CUDA_HOME=/usr/local/cuda-8.0
export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
之后`source /etc/profile`即可

 - 测试
在`终端`输入：
```
>>> nvcc -V
```
会得到相应的nvcc编译器相应的信息，那么CUDA配置成功了。(**记得重启系统**)

如果要进行`cuda性能测试`，可以进行：
```shell
>>> cd /usr/local/cuda/samples
>>> sudo make -j8
```
编译完成后，可以进`samples/bin/.../.../...`的底层目录，运行各类实例。


## 4. 加速库cuDNN（可选）
从官网下载需要注册账号申请，两三天批准。网盘搜索一般也能找到最新版。
Linux目前最新的版本是cudnn V6，但对于tensorflow的预编译版本还不支持这个最近版本，建议采用5.1版本，即是cudnn-8.0-win-x64-v5.1-prod.zip。
下载解压出来是名为cuda的文件夹，里面有bin、include、lib，将三个文件夹复制到安装CUDA的地方覆盖对应文件夹，在终端中输入：
```shell
>>> sudo cp include/cudnn.h /usr/local/cuda/include/
>>> sudo cp lib64/* /usr/local/cuda/lib64/
>>> cd /usr/local/cuda/lib64
>>> sudo ln -sf libcudnn.so.5.1.10 libcudnn.so.5
>>> sudo ln -sf libcudnn.so.5 libcudnn.so
>>> sudo ldconfig -v
```

# Keras框架搭建

## 相关开发包安装
在`终端`中输入:
```shell
>>> sudo pip install -U --pre pip setuptools wheel
>>> sudo pip install -U --pre numpy scipy matplotlib scikit-learn scikit-image
>>> sudo pip install -U --pre tensorflow-gpu
# >>> sudo pip install -U --pre tensorflow ## CPU版本
>>> sudo pip install -U --pre keras
```
安装完毕后，输入`python`，然后输入：
```python
>>> import tensorflow
>>> import keras
```
无错输出即可


## Keras中mnist数据集测试
 下载Keras开发包
```shell
>>> git clone https://github.com/fchollet/keras.git
>>> cd keras/examples/
>>> python mnist_mlp.py
```
程序无错进行，至此，keras安装完成。


## 声明与联系方式 ##

由于作者水平和研究方向所限，无法对所有模块都非常精通，因此文档中不可避免的会出现各种错误、疏漏和不足之处。如果您在使用过程中有任何意见、建议和疑问，欢迎发送邮件到scp173.cool@gmail.com与作者取得联系.

**本教程不得用于任何形式的商业用途，如果需要转载请与作者或中文文档作者联系，如果发现未经允许复制转载，将保留追求其法律责任的权利。**

作者：[SCP-173](https://github.com/KaiwenXiao)
E-mail ：scp173.cool@gmail.com
**如果您需要及时得到指导帮助，可以加微信：SCP173-cool，酌情打赏即可**
![微信](../images/scp_173.png)
