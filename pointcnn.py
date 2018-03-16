
# coding: utf-8

# In[ ]:


import os
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import numpy as np
import time
import math
import random


# In[ ]:


import logging
logging.basicConfig(level=logging.INFO)


# In[ ]:


import collections
import h5py
from transforms3d.euler import euler2mat
import data_utils


# In[ ]:


import mxnet as mx
from mxnet import nd
import mxnet.autograd as ag
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn
from mxnet.gluon.data import Dataset, DataLoader


# In[ ]:


class DotDict(dict):
    """
    Simple class for dot access elements in dict, support nested initialization
    Example:
    d = DotDict({'child': 'dotdict'}, name='dotdict', index=1, contents=['a', 'b'])
    # add new key
    d.new_key = '!' # or d['new_key'] = '!'
    # update values
    d.new_key = '!!!'
    # delete keys
    del d.new_key
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


# In[ ]:


# the returned indices will be used by gather_nd
def get_indices(batch_size, sample_num, point_num, random_sample=True):
    if not isinstance(point_num, np.ndarray):
        point_nums = np.full((batch_size), point_num)
    else:
        point_nums = point_num

    indices = []
    for i in range(batch_size):
        pt_num = point_nums[i]
        if random_sample:
            choices = np.random.choice(pt_num, sample_num, replace=(pt_num < sample_num))
        else:
            choices = np.arange(sample_num) % pt_num
        choices = np.expand_dims(choices, axis=0)
        choices_2d = np.concatenate((np.full_like(choices, i), choices), axis=0)
        indices.append(choices_2d)
    return np.stack(indices, axis=1)


# In[ ]:





# In[ ]:


def gauss_clip(mu, sigma, clip):
    v = random.gauss(mu, sigma)
    v = max(min(v, mu + clip * sigma), mu - clip * sigma)
    return v


# In[ ]:


def uniform(bound):
    return bound * (2 * random.random() - 1)


# In[ ]:


def scaling_factor(scaling_param, method):
    try:
        scaling_list = list(scaling_param)
        return random.choice(scaling_list)
    except:
        if method == 'g':
            return gauss_clip(1.0, scaling_param, 3)
        elif method == 'u':
            return 1.0 + uniform(scaling_param)


# In[ ]:





# In[ ]:


def rotation_angle(rotation_param, method):
    try:
        rotation_list = list(rotation_param)
        return random.choice(rotation_list)
    except:
        if method == 'g':
            return gauss_clip(0.0, rotation_param, 3)
        elif method == 'u':
            return uniform(rotation_param)


# In[ ]:


def get_xforms(xform_num, rotation_range=(0, 0, 0, 'u'), scaling_range=(0.0, 0.0, 0.0, 'u'), order='rxyz'):
    xforms = np.empty(shape=(xform_num, 3, 3))
    rotations = np.empty(shape=(xform_num, 3, 3))
    for i in range(xform_num):
        rx = rotation_angle(rotation_range[0], rotation_range[3])
        ry = rotation_angle(rotation_range[1], rotation_range[3])
        rz = rotation_angle(rotation_range[2], rotation_range[3])
        rotation = euler2mat(rx, ry, rz, order)

        sx = scaling_factor(scaling_range[0], scaling_range[3])
        sy = scaling_factor(scaling_range[1], scaling_range[3])
        sz = scaling_factor(scaling_range[2], scaling_range[3])
        scaling = np.diag([sx, sy, sz])

        xforms[i, :] = scaling * rotation
        rotations[i, :] = rotation
    return xforms, rotations


# In[ ]:





# In[ ]:


def augment(points, xforms, r=None):
    points_xformed = nd.batch_dot(points, xforms, name='points_xformed')
    if r is None:
        return points_xformed

    jitter_data = r * mx.random.normal(shape=points_xformed.shape)
    jitter_clipped = nd.clip(jitter_data, -5 * r, 5 * r, name='jitter_clipped')
    return points_xformed + jitter_clipped


# In[ ]:





# In[ ]:


# a b c
# d e f
# g h i
# a(ei − fh) − b(di − fg) + c(dh − eg)
def compute_determinant(A):
    return A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])            - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])            + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])

# A shape is (N, P, 3, 3)
# return shape is (N, P, 3)
def compute_eigenvals(A):
    A_11 = A[:, :, 0, 0]  # (N, P)
    A_12 = A[:, :, 0, 1]
    A_13 = A[:, :, 0, 2]
    A_22 = A[:, :, 1, 1]
    A_23 = A[:, :, 1, 2]
    A_33 = A[:, :, 2, 2]
    I = nd.eye(3)
    p1 = nd.square(A_12) + nd.square(A_13) + nd.square(A_23)  # (N, P)
    q = (A_11 + A_22 + A_33) / 3  # (N, P)
    p2 = nd.square(A_11 - q) + nd.square(A_22 - q) + nd.square(A_33 - q) + 2 * p1  # (N, P)
    p = nd.sqrt(p2 / 6) + 1e-8  # (N, P)
    N = A.shape[0]
    q_4d = nd.reshape(q, (N, -1, 1, 1))  # (N, P, 1, 1)
    p_4d = nd.reshape(p, (N, -1, 1, 1))
    B = (1 / p_4d) * (A - q_4d * I)  # (N, P, 3, 3)
    r = nd.clip(compute_determinant(B) / 2, -1, 1)  # (N, P)
    phi = nd.arccos(r) / 3  # (N, P)
    eig1 = q + 2 * p * nd.cos(phi)  # (N, P)
    eig3 = q + 2 * p * nd.cos(phi + (2 * math.pi / 3))
    eig2 = 3 * q - eig1 - eig3
    return nd.abs(nd.stack([eig1, eig2, eig3], axis=2)) # (N, P, 3)

# P shape is (N, P, 3), N shape is (N, P, K, 3)
# return shape is (N, P)
def compute_curvature(nn_pts):
    nn_pts_mean = nd.mean(nn_pts, axis=2, keepdims=True)  # (N, P, 1, 3)
    nn_pts_demean = nn_pts - nn_pts_mean  # (N, P, K, 3)
    nn_pts_NPK31 = nd.expand_dims(nn_pts_demean, axis=-1)
    covariance_matrix = nd.batch_dot(nn_pts_NPK31, nn_pts_NPK31, transpose_b=True)  # (N, P, K, 3, 3)
    covariance_matrix_mean = nd.mean(covariance_matrix, axis=2, keepdims=False)  # (N, P, 3, 3)
    eigvals = compute_eigenvals(covariance_matrix_mean)  # (N, P, 3)
    curvature = nd.min(eigvals, axis=-1) / (nd.sum(eigvals, axis=-1) + 1e-8)
    return curvature

def curvature_based_sample(nn_pts, k):
    curvature = compute_curvature(nn_pts)
    point_indices = nd.topk(curvature, axis=-1, k=k, ret_typ='indices')

    pts_shape = nn_pts.shape
    batch_size = pts_shape[0]
    batch_indices = nd.tile(nd.reshape(nd.arange(batch_size), (-1, 1, 1)), (1, k, 1))
    indices = nd.concat(batch_indices, nd.expand_dims(point_indices, axis=2), dim=2)
    return indices


# In[ ]:





# In[ ]:


# input: points(b, n, 3) idx(b, m)
# output: out(b, m, 3)
fwd_source = r'''
    __global__ void gatherpointKernel(int b,int n,int m
        ,const float * __restrict__ inp,const int * __restrict__ idx,float * __restrict__ out){
      for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
          int a=idx[i*m+j];
          out[(i*m+j)*3+0]=inp[(i*n+a)*3+0];
          out[(i*m+j)*3+1]=inp[(i*n+a)*3+1];
          out[(i*m+j)*3+2]=inp[(i*n+a)*3+2];
        }
      }
    }
'''

bwd_source = r'''
    __global__ void scatteraddpointKernel(int b,int n,int m
        ,const float * __restrict__ out_g,const int * __restrict__ idx,float * __restrict__ inp_g){
      for (int i=blockIdx.x;i<b;i+=gridDim.x){
        for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
          int a=idx[i*m+j];
          atomicAdd(&inp_g[(i*n+a)*3+0],out_g[(i*m+j)*3+0]);
          atomicAdd(&inp_g[(i*n+a)*3+1],out_g[(i*m+j)*3+1]);
          atomicAdd(&inp_g[(i*n+a)*3+2],out_g[(i*m+j)*3+2]);
        }
      }
    }        
'''

fwd_module = mx.rtc.CudaModule(fwd_source, exports=['gatherpointKernel'])
fwd_kernel = fwd_module.get_kernel("gatherpointKernel"
                                        , "int b,int n,int m,const float * inp,const int * idx,float * out")

bwd_module = mx.rtc.CudaModule(bwd_source, exports=['scatteraddpointKernel'])
bwd_kernel = bwd_module.get_kernel("scatteraddpointKernel"
                                        , "int b,int n,int m,const float * out_g,const int * idx,float * inp_g")

class GatherPoint(mx.operator.CustomOp):
    def __init__(self):
        super(GatherPoint, self).__init__()


        
    def forward(self, is_train, req, in_data, out_data, aux):
        if req[0] == "null":
            return
        x = in_data[0]  # points
        idx = in_data[1] # idx
        B, N, _ = x.shape
        _, M = idx.shape
        y = mx.nd.empty(shape=(B, M, 3), ctx = x.context, dtype=np.float32) # output

        # args, ctx, grid_shape, block_shape, shared_mem = 0
        fwd_kernel.launch([B, N, M, x, idx, y], x.context, (2, 8, 1), (512, 1, 1))

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = mx.nd.empty(shape=in_data[0].shape, ctx = x.context, dtype=np.float32)
        
        bwd_kernel.launch([B, N, M, out_grad[0], in_data[1], dx], in_data[0].context, (2, 8, 1), (512, 1, 1))
        
        self.assign(in_grad[0], req[0], dx)

@mx.operator.register("GatherPoint")
class GatherPointProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(GatherPointProp, self).__init__(need_top_grad=True)

    def list_arguments(self):
        return ['in_data', 'idx']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        output_shape = (in_shape[1][0], in_shape[1][1], 3)
        return in_shape, [output_shape], []

    def infer_type(self, in_type):
        return in_type, [np.float32], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return GatherPoint()


# In[ ]:


# Input dataset: (b, n, 3), tmp: (b, n)
# Ouput idxs (b, m)
source = r'''
    __global__ void farthestpointsamplingKernel(int b,int n,int m
        ,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
      if (m<=0)
        return;
      const int BlockSize=512;
      __shared__ float dists[BlockSize];
      __shared__ int dists_i[BlockSize];
      const int BufferSize=3072;
      __shared__ float buf[BufferSize*3];
      for (int i=blockIdx.x;i<b;i+=gridDim.x){
        int old=0;
        if (threadIdx.x==0)
          idxs[i*m+0]=old;
        for (int j=threadIdx.x;j<n;j+=blockDim.x){
          temp[blockIdx.x*n+j]=1e38;
        }
        for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
          buf[j]=dataset[i*n*3+j];
        }
        __syncthreads();
        for (int j=1;j<m;j++){
          int besti=0;
          float best=-1;
          float x1=dataset[i*n*3+old*3+0];
          float y1=dataset[i*n*3+old*3+1];
          float z1=dataset[i*n*3+old*3+2];
          for (int k=threadIdx.x;k<n;k+=blockDim.x){
            float td=temp[blockIdx.x*n+k];
            float x2,y2,z2;
            if (k<BufferSize){
              x2=buf[k*3+0];
              y2=buf[k*3+1];
              z2=buf[k*3+2];
            }else{
              x2=dataset[i*n*3+k*3+0];
              y2=dataset[i*n*3+k*3+1];
              z2=dataset[i*n*3+k*3+2];
            }
            float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
            float d2=min(d,td);
            if (d2!=td)
              temp[blockIdx.x*n+k]=d2;
            if (d2>best){
              best=d2;
              besti=k;
            }
          }
          dists[threadIdx.x]=best;
          dists_i[threadIdx.x]=besti;
          for (int u=0;(1<<u)<blockDim.x;u++){
            __syncthreads();
            if (threadIdx.x<(blockDim.x>>(u+1))){
              int i1=(threadIdx.x*2)<<u;
              int i2=(threadIdx.x*2+1)<<u;
              if (dists[i1]<dists[i2]){
                dists[i1]=dists[i2];
                dists_i[i1]=dists_i[i2];
              }
            }
          }
          __syncthreads();
          old=dists_i[0];
          if (threadIdx.x==0)
            idxs[i*m+j]=old;
        }
      }
    }
'''

module = mx.rtc.CudaModule(source, exports=['farthestpointsamplingKernel'])
kernel = module.get_kernel("farthestpointsamplingKernel", "int b,int n,int m,const float * dataset,float * temp,int * idxs")

class FarthestPointSampling(mx.operator.CustomOp):
    def __init__(self, npoints):
        super(FarthestPointSampling, self).__init__()
        self.npoints = npoints
    def forward(self, is_train, req, in_data, out_data, aux):
        if req[0] == "null":
            return
        x = in_data[0]  # input
        B, N, _ = x.shape
        tmp = mx.nd.ones(shape=(B, N), ctx = x.context) * 1e10
        y = mx.nd.empty(shape=(B, self.npoints), ctx = x.context, dtype=np.int32) # output

        # args, ctx, grid_shape, block_shape, shared_mem = 0
        kernel.launch([B, N, self.npoints, x, tmp, y], x.context, (32, 1, 1), (512, 1, 1))

        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register("FarthestPointSampling")
class FarthestPointSamplingProp(mx.operator.CustomOpProp):
    def __init__(self, npoints=0):
        super(FarthestPointSamplingProp, self).__init__(need_top_grad=False)
        
        self.npoints = int(npoints)

    def list_arguments(self):
        return ['in_data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (in_shape[0][0], self.npoints)
        return [data_shape], [output_shape], []

    def infer_type(self, in_type):
        return in_type, [np.int32], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return FarthestPointSampling(self.npoints)


# In[ ]:


def get_shape(x):
    if isinstance(x, mx.nd.NDArray):
        return x.shape
    elif isinstance(x, mx.symbol.Symbol):
        _,x_shape,_=x.infer_shape_partial()
        return x_shape[0] 


# In[ ]:


class BN(nn.HybridBlock):
    def __init__(self):
        super(BN, self).__init__()
        self.bn = nn.BatchNorm(axis=1, use_global_stats=False)
    def hybrid_forward(self, F ,x):
        x = F.transpose(x, axes=(0,3,1,2))
        x = self.bn(x)
        x = F.transpose(x, axes=(0,2,3,1))
        return x
    
class SepCONV(nn.HybridBlock):
    def __init__(self, inp, output, kernel_size, depth_multiplier=1, with_bn=True, activation='elu'):
        super(SepCONV, self).__init__()

        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(channels=int(inp*depth_multiplier), groups=int(inp), kernel_size=kernel_size, strides=(1,1), use_bias=True),
            nn.Conv2D(channels=output, kernel_size=(1,1), strides=(1,1), use_bias=False if with_bn else True)
        )
        self.act = activation
        self.with_bn = with_bn
        if activation is not None:
            self.elu = nn.ELU()
        if with_bn:
            self.bn = nn.BatchNorm(axis=1, use_global_stats=False)
    def hybrid_forward(self, F ,x):
        x = F.transpose(x, axes=(0,3,1,2))
        x = self.net(x)
        if self.act is not None:
            x = self.elu(x)
        if self.with_bn:
            x = self.bn(x)
        x = F.transpose(x, axes=(0,2,3,1))
        return x

class CONV(nn.HybridBlock):
    def __init__(self, output, kernel_size, with_bn=True, activation='elu'):
        super(CONV, self).__init__()
        self.net = nn.Conv2D(channels=output, kernel_size=kernel_size, strides=(1,1), use_bias=False if with_bn else True)
        self.act = activation
        self.with_bn = with_bn
        if activation is not None:
            self.elu = nn.ELU()
        if with_bn:
            self.bn = nn.BatchNorm(axis=1, use_global_stats=False)
    def hybrid_forward(self, F ,x):
        x = F.transpose(x, axes=(0,3,1,2))
        x = self.net(x)
        if self.act is not None:
            x = self.elu(x)
        if self.with_bn:
            x = self.bn(x)
        x = F.transpose(x, axes=(0,2,3,1))
        return x        

class DENSE(nn.HybridBlock):
    def __init__(self, output, drop_rate=0, with_bn=True, activation='elu'):
        super(DENSE, self).__init__()
        self.net = nn.Dense(units=output, flatten=False, use_bias=True)
        self.act = activation
        self.with_bn = with_bn
        self.drop_rate = drop_rate
        if activation is not None:
            self.elu = nn.ELU()
        #if with_bn:
        #    self.bn = nn.BatchNorm(axis=1, use_global_stats=False)
        if drop_rate > 0:
            self.drop = nn.Dropout(drop_rate)
    def hybrid_forward(self, F ,x):
        
        x = self.net(x)
        if self.act is not None:
            x = self.elu(x)
        #if self.with_bn:
        #    x = F.transpose(x, axes=(0,2,1))
        #    x = self.bn(x)
        #    x = F.transpose(x, axes=(0,2,1))
        if self.drop_rate > 0:
            x = self.drop(x)
        
        return x          


# In[ ]:


@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)


# In[ ]:


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
        self.batch_distance_matrix = batch_distance_matrix()
    def hybrid_forward(self, F, points):
        points_shape = get_shape(points)
        batch_size = points_shape[0]
        point_num = points_shape[1]

        D = self.batch_distance_matrix(points)

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
        self.batch_distance_matrix_general = batch_distance_matrix_general()
    def hybrid_forward(self, F, queries, points):
        queries_shape = get_shape(queries)
        batch_size = queries_shape[0]
        point_num = queries_shape[1]

        D = self.batch_distance_matrix_general(queries, points)

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
            scaling_const = F.Variable('scaling_factors', shape=(1,3), init=MyConstant(scaling_factors))
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


# In[ ]:


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
                self.knn_indices_general = knn_indices_general(self.K, False)
            else:
                self.knn_indices_general = knn_indices_general(self.K * self.D, True)
            if self.sorting_method is not None:
                self.sort_points = sort_points(self.sorting_method)
            self.fts_from_pts = nn.HybridSequential()
            self.bn0 = BN()
            self.fts_from_pts.add(
                DENSE(C_pts_fts),
                DENSE(C_pts_fts)
            )

            self.x_trans = nn.HybridSequential()
            self.x_trans.add(
                CONV(K*K, (1, K), with_bn=False),
                DENSE(K*K, with_bn=False),
                DENSE(K*K, with_bn=False, activation=None)
            )
            
            self.sconv0 = SepCONV(C_pts_fts+C_prev, C, (1,K), depth_multiplier)
        
    def hybrid_forward(self, F, pts, fts, qrs):
        if self.D == 1:
            indices = self.knn_indices_general(qrs, pts)
        else:
            indices_dilated = self.knn_indices_general(qrs, pts)
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
            fts_X = F.linalg.gemm2(X, nn_fts_input)
            ###################################################################
        else:
            fts_X = nn_fts_input
        fts = self.sconv0(fts_X)
        return F.squeeze(fts, axis=2)


# In[ ]:





# In[ ]:


class PointCNN(nn.HybridBlock):
    def __init__(self, setting, task, with_feature=True, **kwargs):
        super(PointCNN, self).__init__(**kwargs)
        self.xconv_params = setting.xconv_params
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
            self.xconvs = nn.HybridSequential()
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
                           depth_multiplier, self.sorting_method, prefix="xconv{}_".format(layer_idx) )
                self.xconvs.add(xc)
                
            if self.task == 'segmentation':
                self.xdconvs = nn.HybridSequential()
                self.fuse_fcs = nn.HybridSequential()
                for layer_idx, layer_param in enumerate(setting.xdconv_params):
                    K, D, pts_layer_idx, qrs_layer_idx = layer_param

                    _, _, P, C = self.xconv_params[qrs_layer_idx]
                    _, _, _, C_prev = self.xconv_params[pts_layer_idx]
                    C_pts_fts = C_prev // 4
                    depth_multiplier = 1
                    xdc = xconv(K, D, P, C, C_pts_fts, C_prev, self.with_X_transformation,
                                depth_multiplier, self.sorting_method, prefix="xdconv{}_".format(layer_idx) )
                    self.xdconvs.add(xdc)
                    self.fuse_fcs.add(DENSE(C))

            self.fcs = nn.HybridSequential()       
            for layer_idx, layer_param in enumerate(self.fc_params):
                channel_num, drop_rate = layer_param
                self.fcs.add(DENSE(channel_num, drop_rate))

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
            if P == -1:
                qrs = points
            else:
                if self.with_fps:
                    tmp = F.Custom(pts, name='fps{}_'.format(layer_idx), op_type='FarthestPointSampling', npoints=P)
                    qrs = F.Custom(*[pts, tmp], name='gather{}_'.format(layer_idx), op_type='GatherPoint')
                else:
                    qrs = F.slice(pts, (0, 0, 0), (None, P, None))  # (N, P, 3)
            layer_pts.append(qrs)

            fts_xconv = self.xconvs[layer_idx](pts, fts, qrs)
            layer_fts.append(fts_xconv)
            
        if self.task == 'segmentation':
            for layer_idx, layer_param in enumerate(setting.xdconv_params):
                _, _, pts_layer_idx, qrs_layer_idx = layer_param
                
                pts = layer_pts[pts_layer_idx + 1]
                fts = layer_fts[pts_layer_idx + 1] if layer_idx == 0 else layer_fts[-1]
                qrs = layer_pts[qrs_layer_idx + 1]
                fts_qrs = layer_fts[qrs_layer_idx + 1]
                
                fts_xdconv = self.xdconvs[layer_idx](pts, fts, qrs)
                fts_concat = F.concat(fts_xdconv, fts_qrs, dim=-1)
                fts_fuse = self.fuse_fcs[layer_idx](fts_concat)
                layer_pts.append(qrs)
                layer_fts.append(fts_fuse)
        logits = self.fcs(layer_fts[-1])

        return logits


# In[ ]:





# In[ ]:


setting = DotDict()

setting.num_class = 10

setting.sample_num = 160

setting.batch_size = 256

setting.num_epochs = 2048

setting.step_val = 500

setting.learning_rate_base = 0.01
setting.decay_steps = 8000
setting.decay_rate = 0.6
setting.learning_rate_min = 0.00001

setting.weight_decay = 1e-4

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

setting.optimizer = 'adam'
setting.epsilon = 1e-3

setting.data_dim = 3
setting.with_X_transformation = True
setting.sorting_method = None

setting.keep_remainder = True


# In[ ]:





# In[ ]:


data_train, label_train, data_val, label_val = data_utils.load_cls_train_val('train_files.txt',
                            'test_files.txt')


# In[ ]:


nd_iter = mx.io.NDArrayIter(data={'data': data_train}, label={'softmax_label': label_train}, batch_size=setting.batch_size)


# In[ ]:





# In[ ]:


num_train = data_train.shape[0]
point_num = data_train.shape[1]

batch_num_per_epoch = int(math.ceil(num_train / setting.batch_size))
batch_num = batch_num_per_epoch * setting.num_epochs
batch_size_train = setting.batch_size


# In[ ]:


ctx = [mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3)]
net = PointCNN(setting, 'classification', with_feature=False, prefix="PointCNN_")


# In[ ]:


net.hybridize()


# In[ ]:


def top_1_accuracy(probs, labels, weights=None,is_partial=None, num=None):
    P = probs.asnumpy()
    L = labels.asnumpy()
    W = weights.asnumpy() if weights is not None else None
    if is_partial is not None:
        P = P[0:num, ...] if is_partial else P
        L = L[0:num, ...] if is_partial else L

    #ignore zero weight class
    if W is not None:
        hold_indices = np.greater(W, nd.zeros_like(W))
        probs = P[hold_indices]
        labels = L[hold_indices]

    probs_2d = np.reshape(P, (-1, P.shape[-1]))
    labels_1d = np.reshape(L, [-1])
    labels_1d = labels_1d.astype(np.int64)
    top_1_acc = np.mean(probs_2d.argmax(axis=1) == labels_1d)
    return top_1_acc

def custom_metric(labels, preds):
    return top_1_accuracy(preds, labels)


# In[ ]:


loss=gluon.loss.SoftmaxCrossEntropyLoss(axis=-1)


# In[ ]:





# In[ ]:


sym_max_points = point_num


# In[ ]:


var = mx.sym.var('data', shape=(batch_size_train // len(ctx), sym_max_points, 3))
probs = net(var)

probs_shape = get_shape(probs)
label_var = mx.sym.var('softmax_label', shape=(batch_size_train // len(ctx), probs_shape[1]))


# In[ ]:


def get_loss_sym(probs, labels):
    sm =loss(probs, label_var)
    sm = mx.sym.make_loss(sm, name="softmax")
    res = mx.sym.Group([mx.sym.BlockGrad(probs, name="blockgrad"), sm])
    return res


# In[ ]:


res = get_loss_sym(probs, label_var)


# In[ ]:


mod = mx.mod.Module(res, data_names=['data'], label_names=['softmax_label'], context=ctx)
mod.bind(data_shapes=[('data',(batch_size_train, sym_max_points, 3))]
         , label_shapes=[('softmax_label',(batch_size_train, probs_shape[1]))])
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
mod.init_optimizer(optimizer='sgd', optimizer_params={'learning_rate':0.01, 'momentum': 0.9})


# In[ ]:





# In[ ]:


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
        res = get_loss_sym(probs, label_var)

        labels_tile = nd.tile(labels_2d, (1, probs_shape[1]))
        nb = mx.io.DataBatch(data=[points_sampled], label=[labels_tile], pad=nd_iter.getpad(), index=None)

        mod._symbol = res
        mod.binded=False

        mod.bind(data_shapes=[('data',(batch_size_train, sample_num_train,3))]
                 , label_shapes=[('softmax_label',(batch_size_train, probs_shape[1]))], shared_module=mod
                )

        mod.forward(nb, is_train=True)

        value = custom_metric(nb.label[0], mod.get_outputs()[0])

        mod.backward()
        mod.update()

        t1 = time.time()
        print(ibatch, t1-t0 , value)

