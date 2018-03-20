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
from mxutils import get_shape

from pointcnn import PointCNN, get_indices, get_xforms, augment, custom_metric, get_loss_sym

from dotdict import DotDict
import h5py
import collections
import data_utils

########################### Settings ###############################
setting = DotDict()

setting.num_class = 8

setting.sample_num = 2048

setting.batch_size = 4

setting.num_epochs = 1024

setting.learning_rate_base = 0.001
setting.decay_steps = 20000
setting.decay_rate = 0.9
setting.learning_rate_min = 1e-6

setting.weight_decay = 0.0

setting.jitter = 0.001
setting.jitter_val = 0.0

setting.rotation_range = [0,  0, 0,'g']
setting.rotation_range_val = [0, 0, 0, 'u']

setting.order = 'rxyz'

setting.scaling_range = [0.1, 0.1, 0.1, 'g']
setting.scaling_range_val = [0, 0, 0, 'u']

x = 8

# K, D, P, C
setting.xconv_params = [(8, 1, -1, 32 * x),
                (12, 2, 768, 32 * x),
                (16, 2, 384, 64 * x),
                (16, 6, 128, 128 * x)]

# K, D, pts_layer_idx, qrs_layer_idx
setting.xdconv_params = [(16, 6, 3, 2),
                 (12, 4, 2, 1),
                 (8, 4, 1, 0)]

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

data_train, _ , data_num_train, label_train = data_utils.load_seg('/train_files.txt')
data_val, _ , data_num_val, label_val = data_utils.load_seg('/train_files.txt')

nd_iter = mx.io.NDArrayIter(data={'data': data_train}, label={'softmax_label': label_train}, batch_size=setting.batch_size)
nd_iter_val = mx.io.NDArrayIter(data={'data': data_val}, label={'softmax_label': label_val}, batch_size=setting.batch_size)


num_train = data_train.shape[0]
point_num = data_train.shape[1]

num_val = data_val.shape[0]
val_point_num = data_val.shape[1]

batch_num_per_epoch = int(math.ceil(num_train / setting.batch_size))
batch_num = batch_num_per_epoch * setting.num_epochs
batch_size_train = setting.batch_size

ctx = [mx.gpu(1)]
net = PointCNN(setting, 'segmentation', with_feature=False, prefix="PointCNN_")
net.hybridize()

sym_max_points = setting.sample_num

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

def val_batch(mod,nd_iter_val,setting):
    num_val = (nd_iter_val.data[0][1]).shape[0]
    total_val = 0.0
    iter_size = int(math.ceil(val_point_num*1.0/setting.sample_num))
    
    var = mx.sym.var('data', shape=(iter_size//len(ctx), setting.sample_num, 3))
    probs = net(var)

    probs_shape = get_shape(probs)
    mod._symbol = probs
    mod.binded = False
    mod.bind(data_shapes=[('data',(iter_size//len(ctx), setting.sample_num, 3))], label_shapes=None, shared_module=mod)
    
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
for i in range(5):
    nd_iter.reset()
    for ibatch, batch in enumerate(nd_iter):
        t0 = time.time()

        label = batch.label[0]

        pts_fts = batch.data[0]
        
        points2 = nd.slice(pts_fts, begin=(0,0,0), end= (None, None, 3))
        #features2 = nd.slice(pts_fts, begin=(0,0,3), end= (None, None, None))

        offset = int(random.gauss(0, setting.sample_num // 8))
        offset = max(offset, -setting.sample_num // 4)
        offset = min(offset, setting.sample_num // 4)
        sample_num_train = setting.sample_num + offset

        indices = get_indices(batch_size_train, sample_num_train, point_num)
        indices_nd = nd.array(indices, dtype=np.int32)
        points_sampled = nd.gather_nd(points2, indices=indices_nd)
        labels_sampled = nd.gather_nd(label, indices=indices_nd)
        
        #features_sampled = nd.gather_nd(features2, indices=nd.transpose(indices_nd, (2, 0, 1)))

        #xforms_np, rotations_np = get_xforms(batch_size_train, rotation_range=setting.rotation_range, order=setting.order)
        #points_xformed = nd.batch_dot(points_sampled, nd.array(xforms_np), name='points_xformed')
        #points_augmented = augment(points_sampled, nd.array(xforms_np), setting.jitter)
        features_augmented = None

        #print points_sampled.shape, labels_sampled.shape
        nb = mx.io.DataBatch(data=[points_sampled], label=[labels_sampled], pad=nd_iter.getpad(), index=None)
        #print(nb)

        var = mx.sym.var('data', shape=(batch_size_train // len(ctx), sample_num_train, 3))
        probs = net(var)
        probs_shape = get_shape(probs)
        res = get_loss_sym(probs, label_var)

        mod._symbol = res
        mod.binded=False

        mod.bind(data_shapes=[('data',(batch_size_train, sample_num_train,3))]
                 , label_shapes=[('softmax_label',(batch_size_train, probs_shape[1]))], shared_module=mod
                )

        mod.forward(nb, is_train=True)
        #mod.update_metric(metric1, nb.label)
        #print nb.label[0].shape
        value = custom_metric(nb.label[0], mod.get_outputs()[0])
        
        mod.backward()
        mod.update()
        #name, value = metric1.get()
        t1 = time.time()
        print("iter_num: ",iter_num,  "val: " ,value)
        iter_num = iter_num + 1
    if(((i+1)%5 == 0) and (i > 0)):
        mean_acc = val_batch(mod,nd_iter_val, setting )
        print("Epoch %d: val_mean_acc  %f" %(i,mean_acc))

mod.save_checkpoint("p_seg",400)