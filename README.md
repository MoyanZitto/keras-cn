# keras-cn

本项目由BigMoyan于2016-4-29发起，旨在建立一个[keras.io](keras.io)的中文版文档，并提供更多用户友好的支持与建议。

本项目目前已完成2.x版本，文档网址为[keras-cn](http://keras-cn.readthedocs.io/en/latest/)

如果你在使用keras和keras-cn的过程中有任何问题，请发在issue中，我会定时检查和回复，在issue中你也可以和志同道合的好基友们一起聊天~

项目基于Mkdocs生成静态网页，如果你想为文档做出贡献，请使用Markdown编写文档并遵守以下约定。

## 0.字体颜色

保持默认的字体颜色和字号，锚点的颜色为默认，超链接的颜色为默认
	
当使用```<a name='something'></a>```来设置锚点时，可能引起字体颜色改变，如果颜色发生改变，需要使用font修正字体，默认字体的颜色是#404040

## 1.标题级别

页面大标题为一级，一般每个文件只有一个一级标题 #
页面内的小节是二级标题 ##
小节的小节，如examples里各个example是三级标题 ###
  
## 2.代码块规则

成块的代码使用

\`\`\`python

code

\`\`\`

的形式显式指明
段中代码使用\`\`\`code\`\`\`的形式指明

## 3. 超链接

链接到本项目其他页面的超链接使用相对路径，一个例子是
```[<font color='#FF0000'>text</font>](../models/about_model.md)```
链接到其他外站的链接形式与此相同，只不过圆括号中是绝对地址

## 4.图片

图片保存在docs/images中，插入的例子是：

```

![text](../images/image_name.png)

```

## 5.分割线

每个二级标题之间使用
\*\*\*
产生一个分割线

# 参考网站

## Markdown简明教程

[Markdown](http://wowubuntu.com/markdown/)

## MkDocs中文教程

[MkDocs](http://markdown-docs-zh.readthedocs.io/zh_CN/latest/)

## Keras文档

[Keras](http://keras.io/)

感谢参与！

