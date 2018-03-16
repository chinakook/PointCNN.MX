# coding: utf-8

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'

import math
import random
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import nd
import mxnet.gluon as gluon
from mxutils import get_shape

from pointcnn import PointCNN, get_indices, get_xforms, augment, custom_metric, get_loss_sym

from dotdict import DotDict
import h5py
import collections
import data_utils

########################### Settings ###############################
setting = DotDict()

setting.num_class = 10

setting.sample_num = 160

setting.batch_size = 32

setting.num_epochs = 2048

setting.jitter = 0.01
setting.jitter_val = 0.01

setting.rotation_range = [0, math.pi / 18, 0, 'g']
setting.rotation_range_val = [0, 0, 0, 'u']
setting.order = 'rxyz'

setting.scaling_range = [0.05, 0.05, 0.05, 'g']
setting.scaling_range_val = [0, 0, 0, 'u']

x = 2

# K, D, P, C
setting.xconv_params = [(8, 1, -1, 16 * x),
                (8, 2, -1, 32 * x),
                (8, 4, -1, 48 * x),
                (12, 4, 120, 64 * x),
                (12, 6, 120, 80 * x)]

# C, dropout_rate
setting.fc_params = [(64 * x, 0.0), (32 * x, 0.5)]

setting.with_fps = False

setting.data_dim = 3
setting.with_X_transformation = True
setting.sorting_method = None
###################################################################

data_train, label_train, data_val, label_val = data_utils.load_cls_train_val('train_files.txt',
                            'test_files.txt')

nd_iter = mx.io.NDArrayIter(data={'data': data_train}, label={'softmax_label': label_train}, batch_size=setting.batch_size)


num_train = data_train.shape[0]
point_num = data_train.shape[1]

batch_num_per_epoch = int(math.ceil(num_train / setting.batch_size))
batch_num = batch_num_per_epoch * setting.num_epochs
batch_size_train = setting.batch_size

ctx = [mx.gpu(0)]
net = PointCNN(setting, 'classification', with_feature=False, prefix="PointCNN_")
net.hybridize()

sym_max_points = point_num

var = mx.sym.var('data', shape=(batch_size_train // len(ctx), sym_max_points, 3))

probs = net(var)
probs_shape = get_shape(probs)
label_var = mx.sym.var('softmax_label', shape=(batch_size_train // len(ctx), probs_shape[1]))

loss = get_loss_sym(probs, label_var)

mod = mx.mod.Module(loss, data_names=['data'], label_names=['softmax_label'], context=ctx)
mod.bind(data_shapes=[('data',(batch_size_train, sym_max_points, 3))]
         , label_shapes=[('softmax_label',(batch_size_train, probs_shape[1]))])
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))

mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate':0.01, 'momentum': 0.9})

for i in range(400):
    nd_iter.reset()
    for ibatch, batch in enumerate(nd_iter):
        t0 = time.time()

        label = batch.label[0]
        labels_2d = nd.expand_dims(label,axis=-1)
        pts_fts = batch.data[0]
        bs = pts_fts.shape[0]
        points2 = nd.slice(pts_fts, begin=(0,0,0), end= (None, None, 3))
        #features2 = nd.slice(pts_fts, begin=(0,0,3), end= (None, None, None))

        offset = int(random.gauss(0, setting.sample_num // 8))
        offset = max(offset, -setting.sample_num // 4)
        offset = min(offset, setting.sample_num // 4)
        sample_num_train = setting.sample_num + offset

        indices = get_indices(batch_size_train, sample_num_train, point_num)
        indices_nd = nd.array(indices, dtype=np.int32)
        points_sampled = nd.gather_nd(points2, indices=indices_nd)
        #features_sampled = nd.gather_nd(features2, indices=nd.transpose(indices_nd, (2, 0, 1)))

        xforms_np, rotations_np = get_xforms(batch_size_train, rotation_range=setting.rotation_range, order=setting.order)
        points_xformed = nd.batch_dot(points_sampled, nd.array(xforms_np), name='points_xformed')
        points_augmented = augment(points_sampled, nd.array(xforms_np), setting.jitter)
        features_augmented = None

        var = mx.sym.var('data', shape=(batch_size_train // len(ctx), sample_num_train, 3))
        probs = net(var)
        probs_shape = get_shape(probs)
        loss = get_loss_sym(probs, label_var)

        labels_tile = nd.tile(labels_2d, (1, probs_shape[1]))
        nb = mx.io.DataBatch(data=[points_sampled], label=[labels_tile], pad=nd_iter.getpad(), index=None)

        mod._symbol = loss
        mod.binded=False
        mod.bind(data_shapes=[('data',(batch_size_train, sample_num_train,3))]
                 , label_shapes=[('softmax_label',(batch_size_train, probs_shape[1]))], shared_module=mod)

        mod.forward(nb, is_train=True)

        value = custom_metric(nb.label[0], mod.get_outputs()[0])

        mod.backward()
        mod.update()

        t1 = time.time()
        print(ibatch, t1-t0 , value)

