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

from mxutils import get_shape
from sampleiter import SampleIter
from pointcnn import PointCNN, custom_metric

from dotdict import DotDict
import h5py
import collections
import data_utils

########################### Settings ###############################
setting = DotDict()

setting.num_class = 10

setting.sample_num = 160

setting.batch_size = 256

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

ctx = [mx.gpu(0)]
net = PointCNN(setting, 'classification', with_feature=False, prefix="")
net.hybridize()


data_train, label_train, data_val, label_val = data_utils.load_cls_train_val('./mnist/train_files.txt',
                            './mnist/test_files.txt')

nd_iter = SampleIter(setting=setting, data=data_train, label=label_train, data_pad=data_train.shape[1], batch_size=setting.batch_size, shuffle=True)

#nd_iter_val = mx.io.NDArrayIter(data={'data': data_val}, label={'softmax_label': label_val}, batch_size=setting.batch_size)


num_train = data_train.shape[0]
point_num = data_train.shape[1]

batch_num_per_epoch = int(math.ceil(num_train / setting.batch_size))
batch_num = batch_num_per_epoch * setting.num_epochs
batch_size_train = setting.batch_size

num_val = data_val.shape[0]
batch_num_val = math.floor(num_val / setting.batch_size)

sym_max_points = point_num

var = mx.sym.var('data', shape=(batch_size_train // len(ctx), sym_max_points, 3))

probs = net(var)
probs_shape = get_shape(probs)
label_var = mx.sym.var('softmax_label', shape=(batch_size_train // len(ctx), probs_shape[1]))
loss = mx.sym.SoftmaxOutput(probs, label_var, preserve_shape=True, normalization='valid')



mod = mx.mod.Module(loss, data_names=['data'], label_names=['softmax_label'], context=ctx)
mod.bind(data_shapes=[('data',(batch_size_train, sym_max_points, 3))]
         , label_shapes=[('softmax_label',(batch_size_train, probs_shape[1]))])

mod.init_params(initializer=mx.init.Uniform(0.08))

#lr_sched = mx.lr_scheduler.MultiFactorScheduler([ 200,  400,  600,  800, 1000], 0.33)
mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate':0.4 , 'momentum': 0.9
    , 'wd' : 0.0001, 'clip_gradient': None, 'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0})
    #, 'wd' : 0.0001, 'lr_scheduler': lr_sched, 'clip_gradient': None, 'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0})

def reshape_mod(mod, shape, ctx):
    var = mx.sym.var('data', shape=(shape[0] // len(ctx), shape[1], shape[2]))
    probs = net(var)
    probs_shape = get_shape(probs)
    label_var = mx.sym.var('softmax_label', shape=(shape[0] // len(ctx), probs_shape[1]))

    loss = mx.sym.SoftmaxOutput(probs, label_var, preserve_shape=True, normalization='valid')

    mod._symbol = loss
    mod.binded=False

    mod.bind(data_shapes=[('data', shape)]
                , label_shapes=[('softmax_label',(batch_size_train, probs_shape[1]))], shared_module=mod
            )
    return probs_shape[1]

for i in range(400):
    nd_iter.reset()
    for ibatch, batch in enumerate(nd_iter):
        t0 = time.time()

        points = batch.data[0]
        label = batch.label[0]
        probs_shape = reshape_mod(mod, (batch_size_train, points.shape[1], 3), ctx)

        labels_tile = np.tile(np.expand_dims(label, 1), (1, probs_shape))
        nb = mx.io.DataBatch(data=[nd.array(points)], label=[nd.array(labels_tile)], pad=nd_iter.getpad(), index=None)

        mod.forward(nb, is_train=True)

        value = custom_metric(nd.array(labels_tile), mod.get_outputs()[0])

        mod.backward()
        mod.update()

        t1 = time.time()
        print(ibatch, t1-t0 , value)

    # var = mx.sym.var('data', shape=(setting.batch_size // len(ctx), setting.sample_num, 3))
    # probs = net(var)
    
    # mod._symbol = probs
    # mod.binded=False
    # mod.bind(data_shapes=[('data',(setting.batch_size, setting.sample_num, 3))]
    #             , label_shapes=None, shared_module=mod)

    # nd_iter_val.reset()
    # for ibval, batch_val in enumerate(nd_iter_val):
    #     label = batch_val.label[0]
    #     labels_2d = nd.expand_dims(label,axis=-1)
    #     pts_fts = batch_val.data[0]
    #     bs = pts_fts.shape[0]
    #     points2 = nd.slice(pts_fts, begin=(0,0,0), end= (None, None, 3))
    #     #features2 = nd.slice(pts_fts, begin=(0,0,3), end= (None, None, None))

    #     indices = get_indices(batch_size_train, setting.sample_num, point_num)
    #     indices_nd = nd.array(indices, dtype=np.int32)
    #     points_sampled = nd.gather_nd(points2, indices=indices_nd)
    #     #features_sampled = nd.gather_nd(features2, indices=nd.transpose(indices_nd, (2, 0, 1)))

    #     xforms_np, rotations_np = get_xforms(batch_size_train, rotation_range=setting.rotation_range, order=setting.order)
    #     points_xformed = nd.batch_dot(points_sampled, nd.array(xforms_np), name='points_xformed')
    #     points_augmented = augment(points_sampled, nd.array(xforms_np), setting.jitter)
    #     features_augmented = None

    #     nb = mx.io.DataBatch(data=[points_sampled], label=None, pad=nd_iter_val.getpad(), index=None)

    #     mod.forward(nb, is_train=False)

    #     pred = mod.get_outputs()[0]
    #     pred_mean = nd.mean(pred, axis=1, keepdims=True)

    #     value = custom_metric(labels_2d, pred_mean)

    #     print(ibval, value)