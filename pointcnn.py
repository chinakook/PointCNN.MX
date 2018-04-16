# coding: utf-8

import numpy as np

import math
import random

import mxnet as mx
from mxnet import nd
import mxnet.autograd as ag
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
from mxnet.gluon.data import Dataset, DataLoader

from mxutils import MyConstant, get_shape
from fpsop import *

class BN(nn.HybridBlock):
    def __init__(self):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm(axis=1, use_global_stats=False)
    def hybrid_forward(self, F ,x):
        x = F.transpose(x, axes=(0,3,1,2))
        x = self.bn(x)
        x = F.transpose(x, axes=(0,2,3,1))
        return x

def default_act():
    return nn.Activation('relu')

class SepCONV(nn.HybridBlock):
    def __init__(self, inp, output, kernel_size, depth_multiplier=1, with_bn=True, activation=default_act(), **kwargs):
        super(SepCONV, self).__init__(**kwargs)
        with self.name_scope():
            self.net = nn.HybridSequential()
            cn = int(inp*depth_multiplier)

            if output is None:
                self.net.add(
                    nn.Conv2D(in_channels=inp, channels=cn, groups=inp, kernel_size=kernel_size, strides=(1,1)
                        , use_bias=not with_bn)
                )
            else:
                self.net.add(
                    nn.Conv2D(in_channels=inp, channels=cn, groups=inp, kernel_size=kernel_size, strides=(1,1)
                        , use_bias=True),
                    nn.Conv2D(in_channels=cn, channels=output, kernel_size=(1,1), strides=(1,1)
                        , use_bias=not with_bn)
                )

            self.with_bn = with_bn
            self.act = activation
            if with_bn:
                self.bn = nn.BatchNorm(axis=1, use_global_stats=False)
    def hybrid_forward(self, F ,x):
        x = F.transpose(x, axes=(0,3,1,2))
        x = self.net(x)
        if self.with_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        x = F.transpose(x, axes=(0,2,3,1))
        return x

class CONV(nn.HybridBlock):
    def __init__(self, output, kernel_size, with_bn=True, activation=default_act(), **kwargs):
        super(CONV, self).__init__(**kwargs)
        self.net = nn.Conv2D(channels=output, kernel_size=kernel_size, strides=(1,1)
            ,use_bias=not with_bn)
        self.with_bn = with_bn
        self.act = activation
        if with_bn:
            self.bn = nn.BatchNorm(axis=1, use_global_stats=False)
    def hybrid_forward(self, F ,x):
        x = F.transpose(x, axes=(0,3,1,2))
        x = self.net(x)
        if self.with_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        x = F.transpose(x, axes=(0,2,3,1))
        return x        

class DENSE(nn.HybridBlock):
    def __init__(self, output, drop_rate=0, with_bn=True, activation=default_act()):
        super(DENSE, self).__init__()
        self.net = nn.Dense(units=output, flatten=False, use_bias=not with_bn)
        self.with_bn = with_bn
        self.act = activation
        self.drop_rate = drop_rate
        if with_bn:
            self.bn = nn.BatchNorm(axis=1, use_global_stats=False)
        if self.act is not None and self.drop_rate > 0:
            self.drop = nn.Dropout(drop_rate)
    def hybrid_forward(self, F ,x): 
        x = self.net(x)
        if self.with_bn:
            xl = len(get_shape(x))
            x = F.transpose(x, axes=(0,2,1)) if xl==3 else F.transpose(x, axes=(0,3,1,2))
            x = self.bn(x)
            x = F.transpose(x, axes=(0,2,1)) if xl==3 else F.transpose(x, axes=(0,2,3,1))
        if self.act is not None:
            x = self.act(x)
        if self.act is not None and self.drop_rate > 0:
            x = self.drop(x)        
        return x          

# Init symbol from list
# A shape is (N, C)
class distance_matrix(nn.HybridBlock):
    def __init__(self):
        super(distance_matrix, self).__init__()
    def hybrid_forward(self, F, A):
        r = F.sum(A * A, 1, keepdims=True)
        m = F.batch_dot(A, A, transpose_b=True)
        D = F.broadcast_add(F.broadcast_sub(r, 2 * m), F.transpose(r))
        return D

# A shape is (N, P, C)
class batch_distance_matrix(nn.HybridBlock):
    def __init__(self):
        super(batch_distance_matrix, self).__init__()
    def hybrid_forward(self, F, A):
        r = F.sum(A * A, axis=2, keepdims=True)
        m = F.batch_dot(A, F.transpose(A, axes=(0, 2, 1)))
        D = F.broadcast_add(F.broadcast_sub(r, 2 * m), F.transpose(r, axes=(0, 2, 1)))
        return D

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
class batch_distance_matrix_general(nn.HybridBlock):
    def __init__(self):
        super(batch_distance_matrix_general, self).__init__()
    def hybrid_forward(self, F, A, B):
        r_A = F.sum(A * A, axis=2, keepdims=True)
        r_B = F.sum(B * B, axis=2, keepdims=True)
        m = F.batch_dot(A, F.transpose(B, axes=(0, 2, 1)))
        D = F.broadcast_add(F.broadcast_sub(r_A, 2 * m), F.transpose(r_B, axes=(0, 2, 1)))
        return D

# return shape is (2, N, P, K)
class knn_indices(nn.HybridBlock):
    def __init__(self, k, sort=True):
        super(knn_indices, self).__init__()
        self.k = k
        self.sort = sort
        with self.name_scope():
            self.bdm = batch_distance_matrix()
    def hybrid_forward(self, F, points):
        points_shape = get_shape(points)
        batch_size = points_shape[0]
        point_num = points_shape[1]

        D = self.bdm(points)

        sorttype = False if self.sort else None
        point_indices = F.topk(-D, axis=-1, k=self.k, ret_typ='indices', is_ascend=sorttype)
        batch_indices = F.tile(F.reshape(F.arange(batch_size), (1, -1, 1, 1)), (1, 1, point_num, self.k))
        indices = F.concat(batch_indices, F.expand_dims(point_indices, axis=0), dim=0)
        return indices

# return shape is (2, N, P, K)
class knn_indices_general(nn.HybridBlock):
    def __init__(self, k, sort=True):
        super(knn_indices_general, self).__init__()
        self.k = k
        self.sort = sort
        with self.name_scope():
            self.bdmg = batch_distance_matrix_general()
    def hybrid_forward(self, F, queries, points):
        queries_shape = get_shape(queries)
        batch_size = queries_shape[0]
        point_num = queries_shape[1]

        D = self.bdmg(queries, points)

        sorttype = False if self.sort else True
        point_indices = F.topk(-D, axis=-1, k=self.k, ret_typ='indices', is_ascend=sorttype)  # (N, P, K)
        batch_indices = F.tile(F.reshape(F.arange(batch_size), (1, -1, 1, 1)), (1, 1, point_num, self.k))
        indices = F.concat(batch_indices, F.expand_dims(point_indices, axis=0), dim=0)
        return indices

# indices is (2, N, P, K)
# return shape is (2, N, P, K)
class sort_points(nn.HybridBlock):
    def __init__(self, sorting_method):
        super(sort_points, self).__init__()
        self.sorting_method = sorting_method
        if sorting_method.startswith('c'):
            if ''.join(sorted(sorting_method[1:])) != 'xyz':
                print('Unknown sorting method!')
                exit()
            self.epsilon = 1e-8
            self.scaling_factors = [math.pow(100.0, 3 - sorting_method.find('x')),
                               math.pow(100.0, 3 - sorting_method.find('y')),
                               math.pow(100.0, 3 - sorting_method.find('z'))]
            
        elif self.sorting_method == 'l2':
            pass
        else:
            print('Unknown sorting method!')
            exit()        
        
    def hybrid_forward(self, F, points, indices):
        indices_shape = get_shape(indices)
        batch_size = indices_shape[1]
        point_num = indices_shape[2]
        k = indices_shape[3]

        nn_pts = F.gather_nd(points, indices)  # (N, P, K, 3)
        if self.sorting_method.startswith('c'):
            nn_pts_min = F.min(nn_pts, axis=2, keepdims=True)
            nn_pts_max = F.max(nn_pts, axis=2, keepdims=True)
            nn_pts_normalized = (nn_pts - nn_pts_min) / (nn_pts_max - nn_pts_min + self.epsilon)  # (N, P, K, 3)
            scaling_const = F.Variable('scaling_factors', shape=(1,3), init=MyConstant(self.scaling_factors))
            scaling_const = F.BlockGrad(scaling_const)
            scaling = F.reshape(scaling_const, (1,1,1,3))
            sorting_data = F.sum(nn_pts_normalized * scaling, axis=-1, keepdims=False)  # (N, P, K)
        elif self.sorting_method == 'l2':
            nn_pts_center = F.mean(nn_pts, axis=2, keepdims=True)  # (N, P, 1, 3)
            nn_pts_local = F.broadcast_sub(nn_pts, nn_pts_center)  # (N, P, K, 3)
            #sorting_data = norm(nn_pts_local, axis=-1, keep_dims=False)  # (N, P, K)
            sorting_data = F.sqrt(F.sum(F.multiply(nn_pts_local, nn_pts_local),axis=-1, keepdims=False))

        k_indices = F.topk(sorting_data, axis=-1, k=k, ret_typ='indices', is_ascend=False)  # (N, P, K)
        batch_indices = F.tile(F.reshape(F.arange(batch_size), (1,-1, 1, 1)), (1, 1, point_num, k))
        point_indices = F.tile(F.reshape(F.arange(point_num), (1, 1, -1, 1)), (1, batch_size, 1, k))
        k_indices_4d = F.expand_dims(k_indices, axis=0)
        sorting_indices = F.concat(batch_indices, point_indices, k_indices_4d, dim=0)  # (N, P, K, 3)
        return F.transpose(F.gather_nd(F.transpose(indices, axes=(1,2,3,0)), sorting_indices), axes=(3,0,1,2))

def top_1_accuracy(probs, labels, weights=None,is_partial=None, num=None):
    P = probs.asnumpy()
    L = labels.asnumpy()
    W = weights.asnumpy() if weights is not None else None
    if is_partial is not None:
        P = P[0:num, ...] if is_partial else P
        L = L[0:num, ...] if is_partial else L

    #ignore zero weight class
    if W is not None:
        hold_indices = np.greater(W, np.zeros_like(W))
        probs = P[hold_indices]
        labels = L[hold_indices]

    probs_2d = np.reshape(P, (-1, P.shape[-1]))
    labels_1d = np.reshape(L, [-1])
    labels_1d = labels_1d.astype(np.int64)
    top_1_acc = np.mean(probs_2d.argmax(axis=1) == labels_1d)
    return top_1_acc

def custom_metric(labels, preds):
    return top_1_accuracy(preds, labels)

class xconv(nn.HybridBlock):
    def __init__(self, K, D, P, C, C_pts_fts, C_prev, with_X_transformation, depth_multiplier
                 ,sorting_method=None, **kwargs):
        super(xconv, self).__init__(**kwargs)
        self.K = K
        self.D = D
        self.P = P
        self.C = C
        self.with_X_transformation = with_X_transformation
        self.depth_multiplier = depth_multiplier
        self.sorting_method = sorting_method
        with self.name_scope():
            if self.D == 1:
                self.kig = knn_indices_general(self.K, False)
            else:
                self.kig = knn_indices_general(self.K * self.D, True)
            if self.sorting_method is not None:
                self.sort_points = sort_points(self.sorting_method)
            self.fts_from_pts = nn.HybridSequential()
            self.bn0 = BN()
            self.fts_from_pts.add(
                DENSE(C_pts_fts, activation=nn.PReLU()),
                DENSE(C_pts_fts, activation=nn.PReLU())
            )
            if self.with_X_transformation:
                # self.trans0 = CONV(K*K, (1, K))
                # self.trans1 = SepCONV(K, output=None, kernel_size=(1,K), depth_multiplier=K)
                # self.trans2 = SepCONV(K, output=None, kernel_size=(1,K), depth_multiplier=K)
                self.x_trans = nn.HybridSequential()
                self.x_trans.add(
                    CONV(K*K, (1, K), with_bn=False, activation=nn.PReLU()),
                    DENSE(K*K, with_bn=False, activation=nn.PReLU()),
                    DENSE(K*K, with_bn=False, activation=None)
                )
            
            #self.sconv0 = CONV(C, (1,K))
            self.sconv0 = SepCONV(C_pts_fts+C_prev, C, (1,K), depth_multiplier, prefix="fts_")
        
    def hybrid_forward(self, F, pts, fts, qrs):
        #print(get_shape(pts), get_shape(qrs))
        if self.D == 1:
            indices = self.kig(qrs, pts)
        else:
            indices_dilated = self.kig(qrs, pts)
            indices = F.slice(indices_dilated, begin=(0,0,0,0), end=(None,None,None,None), step=(None,None,None,self.D))

        P = get_shape(qrs)[1] if self.P == -1 else self.P
        if self.sorting_method is not None:
            indices = self.sort_points(pts, indices)

        nn_pts = F.gather_nd(pts, indices)  # (N, P, K, 3)
        nn_pts_center = F.expand_dims(qrs, axis=2)  # (N, P, 1, 3)
        nn_pts_local = F.broadcast_sub(nn_pts, nn_pts_center)  # (N, P, K, 3)
        
        # Prepare features to be transformed
        nn_pts_local_bn = self.bn0(nn_pts_local)
        nn_fts_from_pts = self.fts_from_pts(nn_pts_local_bn)

        if fts is None:
            nn_fts_input = nn_fts_from_pts
        else:
            nn_fts_from_prev = F.gather_nd(fts, indices)
            nn_fts_input = F.concat(nn_fts_from_pts, nn_fts_from_prev, dim=-1)

        if self.with_X_transformation:
            ######################## X-transformation #########################
            X_2 = self.x_trans(nn_pts_local_bn)
            X = F.reshape(X_2, (-1, P, self.K, self.K))
            # X_0 = self.trans0(nn_pts_local_bn)
            # X_0_KK = F.reshape(X_0, (-1, P, self.K, self.K))
            # X_1 = self.trans1(X_0_KK)
            # X_1_KK = F.reshape(X_1, (-1, P, self.K, self.K))
            # X_2 = self.trans2(X_1_KK)
            # X_2_KK = F.reshape(X_2, (-1, P, self.K, self.K))
            fts_X = F.linalg.gemm2(X, nn_fts_input)
            ###################################################################
        else:
            fts_X = nn_fts_input
        fts = self.sconv0(fts_X)
        return F.squeeze(fts, axis=2)

class PointCNN(nn.HybridBlock):
    def __init__(self, setting, task, with_feature=True, **kwargs):
        super(PointCNN, self).__init__(**kwargs)
        self.xconv_params = setting.xconv_params
        self.xdconv_params = setting.xdconv_params
        self.fc_params = setting.fc_params
        self.with_X_transformation = setting.with_X_transformation
        self.sorting_method = setting.sorting_method
        self.num_class = setting.num_class
        self.with_fps = setting.with_fps
        self.task = task
        self.with_feature = with_feature

        with self.name_scope():
            if with_feature:
                C_fts = self.xconv_params[0][-1] // 2
                self.dense0 = DENSE(C_fts)
            self.xconvs = nn.HybridBlock()
            for layer_idx, layer_param in enumerate(self.xconv_params):
                K, D, P, C = layer_param

                if layer_idx == 0:
                    C_prev = 0
                    C_pts_fts = C // 4 if with_feature else C // 2
                    depth_multiplier = 4
                else:
                    C_prev = self.xconv_params[layer_idx - 1][-1]
                    C_pts_fts = C_prev // 4
                    depth_multiplier = math.ceil(C / C_prev)
                xc = xconv(K, D, P, C, C_pts_fts, C_prev, self.with_X_transformation,
                           depth_multiplier, self.sorting_method, prefix="xconv{}_".format(layer_idx+1) )
                self.xconvs.register_child(xc)
                
            if self.task == 'segmentation':
                self.xdconvs = nn.HybridBlock()
                self.fuse_fcs = nn.HybridBlock(prefix="fts_fuse_")
                for layer_idx, layer_param in enumerate(self.xdconv_params):
                    K, D, pts_layer_idx, qrs_layer_idx = layer_param

                    _, _, P, C = self.xconv_params[qrs_layer_idx]
                    _, _, _, C_prev = self.xconv_params[pts_layer_idx]
                    C_pts_fts = C_prev // 4
                    depth_multiplier = 1
                    xdc = xconv(K, D, P, C, C_pts_fts, C_prev, self.with_X_transformation,
                                depth_multiplier, self.sorting_method, prefix="xdconv{}_".format(layer_idx+1) )
                    self.xdconvs.register_child(xdc)
                    with self.fuse_fcs.name_scope():
                        self.fuse_fcs.register_child(DENSE(C))

            self.fcs = nn.HybridSequential(prefix="fc_")
            with self.fcs.name_scope():       
                for layer_idx, layer_param in enumerate(self.fc_params):
                    channel_num, drop_rate = layer_param
                    self.fcs.add(DENSE(channel_num, drop_rate))
            with self.fcs.name_scope(): 
                self.fcs.add(DENSE(self.num_class, with_bn=False, activation=None))
        
    def hybrid_forward(self, F, points, features=None):
        layer_pts = [points]
        if self.with_feature and features is not None:
            features = self.dense0(features)
        layer_fts = [features]

        for layer_idx, layer_param in enumerate(self.xconv_params):
            P = layer_param[2]
            pts = layer_pts[-1]
            fts = layer_fts[-1]
            if P == -1 or (layer_idx > 0 and P == self.xconv_params[layer_idx-1][2]):
                qrs = layer_pts[-1]
            else:
                if self.with_fps:
                    idx = F.Custom(pts, op_type='FarthestPointSampling', npoints=P)
                    qrs = F.Custom(*[pts, idx], op_type='GatherPoint')
                else:
                    qrs = F.slice(pts, begin=(0, 0, 0), end=(None, P, None))  # (N, P, 3)
            layer_pts.append(qrs)
            
            fts_xconv = self.xconvs._children[layer_idx](pts, fts, qrs)
            layer_fts.append(fts_xconv)
            
        if self.task == 'segmentation':
            for layer_idx, layer_param in enumerate(self.xdconv_params):
                _, _, pts_layer_idx, qrs_layer_idx = layer_param
                
                pts = layer_pts[pts_layer_idx + 1]
                fts = layer_fts[pts_layer_idx + 1] if layer_idx == 0 else layer_fts[-1]
                qrs = layer_pts[qrs_layer_idx + 1]
                fts_qrs = layer_fts[qrs_layer_idx + 1]

                fts_xdconv = self.xdconvs._children[layer_idx](pts, fts, qrs)
                fts_concat = F.concat(fts_xdconv, fts_qrs, dim=-1)
                fts_fuse = self.fuse_fcs._children[layer_idx](fts_concat)
                layer_pts.append(qrs)
                layer_fts.append(fts_fuse)
        logits = self.fcs(layer_fts[-1])

        return logits

if __name__ == "__main__":
    pass
