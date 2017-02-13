# Keras使用陷阱

这里归纳了Keras使用过程中的一些常见陷阱和解决方法，如果你的模型怎么调都搞不对，或许你有必要看看是不是掉进了哪个猎人的陷阱，成为了一只嗷嗷待宰（？）的猎物

Keras陷阱不多，我们保持更新，希望能做一个陷阱大全

内有恶犬，小心哟

## TF卷积核与TH卷积核

Keras提供了两套后端，Theano和Tensorflow，这是一件幸福的事，就像手中拿着馒头，想蘸红糖蘸红糖，想蘸白糖蘸白糖

如果你从无到有搭建自己的一套网络，则大可放心。但如果你想使用一个已有网络，或把一个用th/tf
训练的网络以另一种后端应用，在载入的时候你就应该特别小心了。

卷积核与所使用的后端不匹配，不会报任何错误，因为它们的shape是完全一致的，没有方法能够检测出这种错误。

在使用预训练模型时，一个建议是首先找一些测试样本，看看模型的表现是否与预计的一致。

如需对卷积核进行转换，可以使用utils.np_utils.kernel_convert，或使用utils.layer_utils.convert_all_kernels_in_model来对模型的所有卷积核进行转换

## 向BN层中载入权重
如果你不知道从哪里淘来一个预训练好的BN层，想把它的权重载入到Keras中，要小心参数的载入顺序。

一个典型的例子是，将caffe的BN层参数载入Keras中，caffe的BN由两部分构成，bn层的参数是mean，std，scale层的参数是gamma，beta

按照BN的文章顺序，似乎载入Keras BN层的参数应该是[mean, std, gamma, beta]

然而不是的，Keras的BN层参数顺序应该是[gamma, beta, mean, std]，这是因为gamma和beta是可训练的参数，而mean和std不是

Keras的可训练参数在前，不可训练参数在后

错误的权重顺序不会引起任何报错，因为它们的shape完全相同

## shuffle和validation_split的顺序

模型的fit函数有两个参数，shuffle用于将数据打乱，validation_split用于在没有提供验证集的时候，按一定比例从训练集中取出一部分作为验证集

这里有个陷阱是，程序是先执行validation_split，再执行shuffle的，所以会出现这种情况：

假如你的训练集是有序的，比方说正样本在前负样本在后，又设置了validation_split，那么你的验证集中很可能将全部是负样本

同样的，这个东西不会有任何错误报出来，因为Keras不可能知道你的数据有没有经过shuffle，保险起见如果你的数据是没shuffle过的，最好手动shuffle一下

## merge 和 Merge

Keras有两个很类似的用于张量融合的工具，在这里加以辨析。

merge是一个函数，接受一个tensor列表，并将列表中的tensor按照给定的方式融合在一起形成输出，多用于以Model作为模型的情况

Merge是一个层对象，它接收一个层对象的列表，并按照给定的方式将它们的输出tensor融合起来。典型的使用场景是在Sequential中处理多输入的情况



## 未完待续

如果你在使用Keras中遇到难以察觉的陷阱，请发信到moyan_work@foxmail.com说明~赠人玫瑰，手有余香，前人踩坑，后人沾光，有道是我不入地狱谁入地狱，愿各位Keras使用者积极贡献Keras陷阱。老规矩，陷阱贡献者将被列入致谢一栏
