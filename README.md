# PointCNN.MX
This is a MXNet implementation of [PointCNN](https://github.com/yangyanli/PointCNN). It is as efficent as the origin Tensorflow implemetation and achieves same accuracy on both classification and segmentaion jobs. See the following references for more information:
```
"PointCNN"
Yangyan Li, Rui Bu, Mingchao Sun, Baoquan Chen
arXiv preprint arXiv:1801.07791, 2018.
```
[https://arxiv.org/abs/1801.07791](https://arxiv.org/abs/1801.07791)


# Usage
We've tested code on MNIST only.

```python
python ./download_datasets.py -d mnist -f ./
python ./prepare_mnist_data.py -f ./
python ./pointcnn.py
```

# License
Our code is released under MIT License (see LICENSE file for details).
