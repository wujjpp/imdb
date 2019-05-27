# Deep learning based on Keras - 二分类问题

## 环境准备

- [Tensorflow安装说明](https://tensorflow.google.cn/install/pip)

- [Keras安装说明](https://keras.io/#installation)

- [下载数据文件](http://mng.bz/0tIo)

- 根据实际情况修改data_loader.py中相关配置

```python
# 数据文件解开后存放路径
imbd_dir = '/home/jp/workspace/aclImdb'
train_dir = os.path.join(imbd_dir, 'train')
test_dir = os.path.join(imbd_dir, 'test')

# 下面两个文件是解析数据生成的，注意修改
one_hot_data_file_name = '/home/jp/workspace/imdb-one-hot.npz'
embedding_data_file_name = '/home/jp/workspace/imdb-embedding.npz'
```

## 运行

```shell
$ python dense.py
```

```shell
$ python one_hot.py
```

```shell
$ python embedding.py
```

```shell
$ python rnn.py
```

```shell
$ python lstm.py
```