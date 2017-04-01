# Keras 示例程序


## Keras示例程序


* addition_rnn.py: 序列到序列学习, 实现两个数的加法

* antirectifier.py: 展示了如何在Keras中定制自己的层

* babi_memnn.py: 在bAbI数据集上训练一个记忆网络,用于阅读理解

* babi_rnn.py: 在bAbI数据集上训练一个循环网络,用于阅读理解

* cifar10_cnn.py: 在CIFAR10数据集上训练一个简单的深度CNN网络,用于小图片识别

* conv_filter_visualization.py: 通过在输入空间上梯度上升可视化VGG16的滤波器

* conv_lstm.py: 展示了一个卷积LSTM网络的应用

* deep_dream.py: Google DeepDream的Keras实现

* image_ocr.py:训练了一个卷积+循环网络+CTC logloss来进行OCR

* imdb_bidirectional_lstm.py: 在IMDB数据集上训练一个双向LSTM网络,用于情感分类.

* imdb_cnn.py: 展示了如何在文本分类上如何使用Covolution1D

* imdb_cnn_lstm.py: 训练了一个栈式的卷积网络+循环网络进行IMDB情感分类.

* imdb_fasttext.py: 训练了一个FastText模型用于IMDB情感分类

* imdb_lstm.py: 训练了一个LSTM网络用于IMDB情感分类.

* lstm_benchmark.py: 在IMDB情感分类上比较了LSTM的不同实现的性能

* lstm_text_generation.py: 从尼采的作品中生成文本

* mnist_acgan.py：AC-GAN(Auxiliary Classifier GAN)实现的示例

* mnist_cnn.py: 训练一个用于mnist数据集识别的卷积神经网络

* mnist_hierarchical_rnn.py: 训练了一个HRNN网络用于MNIST数字识别

* mnist_irnn.py: 重现了基于逐像素点序列的IRNN实验,文章见Le et al. "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"

* mnist_mlp.py: 训练了一个简单的多层感知器用于MNIST分类

* mnist_net2net.py: 在mnist上重现了文章中的Net2Net实验,文章为"Net2Net: Accelerating Learning via Knowledge Transfer".

* mnist_siamese_graph.py:基于MNIST训练了一个多层感知器的Siamese网络

* mnist_sklearn_wrapper.py: 展示了如何使用sklearn包装器

* mnist_swwae.py: 基于残差网络和MNIST训练了一个栈式的What-Where自动编码器

* mnist_transfer_cnn.py: 迁移学习的小例子

* neural_doodle.py:神经网络绘画

* neural_style_transfer.py: 图像风格转移

* pretrained_word_embeddings.py: 将GloVe嵌入层载入固化的Keras Embedding层中, 并用以在新闻数据集上训练文本分类模型

* reuters_mlp.py: 训练并评估一个简单的多层感知器进行路透社新闻主题分类

* stateful_lstm.py: 展示了如何使用状态RNN对长序列进行建模

* variational_autoencoder.py: 展示了如何搭建变分编码器

* variational_autoencoder_deconv.py Demonstrates how to build a variational autoencoder with Keras using deconvolution layers.
