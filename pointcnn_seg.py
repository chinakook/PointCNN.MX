# coding: utf-8

import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'

import math
import random
import numpy as np
import time
import logging
logging.basicConfig(level=logging.INFO)
import sys
import mxnet as mx
from mxnet import nd
import mxnet.gluon as gluon

from dotdict import DotDict
import h5py
import collections
import data_utils

from mxutils import get_shape
from sampleiter import SampleIter
from pointcnn import PointCNN, custom_metric

from kktools.utils import statparams

########################### Settings ###############################
setting = DotDict()

setting.num_class = 2

setting.sample_num = 4096

setting.batch_size = 8

setting.num_epochs = 1024

setting.jitter = 0.001
setting.jitter_val = 0.0

setting.rotation_range = [0,  0, 0,'g']
setting.rotation_range_val = [0, 0, 0, 'u']

setting.order = 'rxyz'

setting.scaling_range = [0, 0, 0, 'g']
setting.scaling_range_val = [0, 0, 0, 'u']

x = 8

# K, D, P, C
setting.xconv_params = [(8, 1, -1, 32 * x),
                (12, 2, 768, 32 * x),
                (16, 2, 384, 64 * x),
                (16, 6, 128, 128 * x)]

# K, D, pts_layer_idx, qrs_layer_idx
setting.xdconv_params =  [(16, 6, 3, 2),
                 (12, 6, 2, 1),
                 (8, 6, 1, 0),
                 (8, 4, 0, 0)]


# C, dropout_rate
setting.fc_params = [(32 * x, 0.5), (32 * x, 0.5)]

setting.with_fps = True

setting.data_dim = 3

# changed
setting.use_extra_features = False
# changed
setting.with_normal_feature = False
setting.with_X_transformation = True
setting.sorting_method = None

setting.keep_remainder = True
###################################################################

#data_train, _ , data_num_train, label_train = data_utils.load_seg('/train_files.txt')
#data_val, _ , data_num_val, label_val = data_utils.load_seg('/train_files.txt')
data_train, _ , data_num_train, label_train = data_utils.load_seg('/mnt/15F1B72E1A7798FD/Dataset/point_cnn/label_las/train/2h5/train_files.txt')
data_val, _ , data_num_val, label_val = data_utils.load_seg('/mnt/15F1B72E1A7798FD/Dataset/point_cnn/label_las/val/h5/train_files.txt')

nd_iter = SampleIter(setting=setting, data=data_train, label=label_train, data_pad=data_num_train, batch_size=setting.batch_size, shuffle=True)
# nd_iter_val = SampleIter(setting=setting, data=data_val, label=label_val, data_pad=data_num_val, batch_size=setting.batch_size)

# for batch in nd_iter:
#     print(batch)

num_train = data_train.shape[0]
point_num = data_train.shape[1]

num_val = data_val.shape[0]


batch_num_per_epoch = int(math.ceil(num_train / setting.batch_size))
batch_num = batch_num_per_epoch * setting.num_epochs
batch_size_train = setting.batch_size

ctx = [mx.gpu(0), mx.gpu(1), mx.gpu(2), mx.gpu(3)]
net = PointCNN(setting, 'segmentation', with_feature=False, prefix="")
net.hybridize()

sym_max_points = max(data_train.shape[1], data_val.shape[1])

var = mx.sym.var('data', shape=(batch_size_train // len(ctx), sym_max_points, 3))
probs = net(var)

probs_shape = get_shape(probs)
label_var = mx.sym.var('softmax_label', shape=(batch_size_train // len(ctx), probs_shape[1]))
loss = mx.sym.SoftmaxOutput(probs, label_var, preserve_shape=True, normalization='valid')

# paramscount = statparams(loss, data=(batch_size_train // len(ctx), sym_max_points, 3)
#     , softmax_label = (batch_size_train // len(ctx), probs_shape[1]))
# print(paramscount)

#mx.viz.print_summary(probs, shape={'data':(batch_size_train // len(ctx), sym_max_points, 3)})
#mx.viz.plot_network(probs).view()

mod = mx.mod.Module(loss, data_names=['data'], label_names=['softmax_label'], context=ctx)
mod.bind(data_shapes=[('data',(batch_size_train, sym_max_points, 3))]
         , label_shapes=[('softmax_label',(batch_size_train, probs_shape[1]))])

mod.init_params(initializer=mx.init.Uniform(0.08))
# mod.init_params(initializer=mx.init.Xavier())

lr_sched = mx.lr_scheduler.MultiFactorScheduler([ 200,  400,  600,  800, 1000], 0.33)
mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate':0.4 , 'momentum': 0.9
    , 'wd' : 0.0001, 'lr_scheduler': lr_sched, 'clip_gradient': None, 'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0})

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

def val_batch(mod,nd_iter_val,setting):
    num_val = (nd_iter_val.data[0][1]).shape[0]
    total_val = 0.0
    val_point_num = data_val.shape[1]
    iter_size = int(math.ceil(val_point_num*1.0/setting.sample_num))
    

    reshape_mod(mod, (iter_size//len(ctx), setting.sample_num, 3), ctx)
    
    indices_batch_indices = np.tile(np.reshape(np.arange(iter_size), (iter_size, 1, 1)), (1, setting.sample_num, 1))
    
    for ibval in range(num_val):
        
        label = (nd_iter_val.label[0][1][ibval]).asnumpy()

        pts_fts = nd_iter_val.data[0][1][ibval]
        pts_fts = nd.expand_dims(pts_fts,axis=0)
        
        
        point_num = data_num_val[ibval]
        bs = pts_fts.shape[0]
        points2 = nd.slice(pts_fts, begin=(0,0,0), end= (None, None, 3))

        points_batch = (points2.asnumpy())[[0]*iter_size, ...]
      
        tile_num = int(math.ceil((setting.sample_num * iter_size*1.0) / point_num))
        
        indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:setting.sample_num * iter_size]
        np.random.shuffle(indices_shuffle)
        indices_batch_shuffle = np.reshape(indices_shuffle, (iter_size, setting.sample_num, 1))        
        indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

        data_f = nd.array(points_batch)
        indices_f = nd.array(indices_batch)
        indices_f = nd.transpose(indices_f,(2,0,1))        

        data_in = nd.gather_nd(data_f,indices= indices_f)        
        
        nb = mx.io.DataBatch(data=[data_in], label=None, pad=None, index=None)    

        mod.forward(nb,is_train = False)        
        seg_probs = mod.get_outputs()[0].asnumpy()       
        probs_2d = np.reshape(seg_probs, (setting.sample_num * iter_size, -1))
        predictions = np.zeros(shape=(point_num,))    
        for idx in range(setting.sample_num * iter_size):
            point_idx = indices_shuffle[idx]
            point_probs = probs_2d[idx, :]            
            seg_idx = np.argmax(point_probs)            
            predictions[point_idx] = seg_idx
        ac_lab = label[0:point_num]
        
        acc = (predictions == ac_lab).sum() * 1.0 / point_num   
        
        total_val = total_val + acc

    return total_val*1.0/num_val

#profiler.set_state('run')
iter_num = 0
for i in range(160):
    nd_iter.reset()
    for ibatch, batch in enumerate(nd_iter):
        t0 = time.time()

        points = batch.data[0]
        label = batch.label[0]

        nb = mx.io.DataBatch(data=[points], label=[label], pad=nd_iter.getpad(), index=None)

        reshape_mod(mod, (batch_size_train, points.shape[1], 3), ctx)

        mod.forward(nb, is_train=True)
        #mod.update_metric(metric1, nb.label)
        value = custom_metric(label, mod.get_outputs()[0])
        
        mod.backward()
        mod.update()
        #name, value = metric1.get()
        t1 = time.time()
        print("iter: ",iter_num,  "acc: " ,value, "time: ", t1 - t0)
        iter_num = iter_num + 1
    # if(((i+1)%5 == 0) and (i > 0)):
    #     mean_acc = val_batch(mod,nd_iter_val, setting )
    #     print("Epoch %d: val_mean_acc  %f" %(i,mean_acc))


reshape_mod(mod, (batch_size_train, setting.sample_num, 3), ctx)

mod.save_checkpoint("p_seg", 402)