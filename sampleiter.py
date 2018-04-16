import random
import numpy as np
from transforms3d.euler import euler2mat

import mxnet as mx
from mxnet import ndarray as nd
from mxnet.ndarray import array
from mxnet.io import DataIter
from mxnet.io import DataDesc, DataBatch

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

def gauss_clip(mu, sigma, clip):
    v = random.gauss(mu, sigma)
    v = max(min(v, mu + clip * sigma), mu - clip * sigma)
    return v

def uniform(bound):
    return bound * (2 * random.random() - 1)

def scaling_factor(scaling_param, method):
    try:
        scaling_list = list(scaling_param)
        return random.choice(scaling_list)
    except:
        if method == 'g':
            return gauss_clip(1.0, scaling_param, 3)
        elif method == 'u':
            return 1.0 + uniform(scaling_param)

def rotation_angle(rotation_param, method):
    try:
        rotation_list = list(rotation_param)
        return random.choice(rotation_list)
    except:
        if method == 'g':
            return gauss_clip(0.0, rotation_param, 3)
        elif method == 'u':
            return uniform(rotation_param)

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

def augment(points, xforms, r=None):
    points_xformed = nd.batch_dot(points, xforms)
    if r is None:
        return points_xformed

    jitter_data = r * mx.random.normal(shape=points_xformed.shape)
    jitter_clipped = nd.clip(jitter_data, -5 * r, 5 * r)
    return points_xformed + jitter_clipped

class SampleIter(DataIter):
    def __init__(self, setting, data, label=None, data_pad=None, batch_size=1, shuffle=False,
                 last_batch_handle='pad', data_name='data',
                 label_name='softmax_label'):
        super(SampleIter, self).__init__(batch_size)

        self.data = data
        self.label = label
        self.data_pad = data_pad
        self.max_point_num = data.shape[1]
        self.feat_dim = data.shape[2]

        self.setting = setting

        # batching
        # if last_batch_handle == 'discard':
        #     new_n = self.data[0][1].shape[0] - self.data[0][1].shape[0] % batch_size
        #     self.idx = self.idx[:new_n]

        self.num_data = data.shape[0]
        self.shuffle = shuffle
        if shuffle:
            self.idx = list(range(0, self.num_data))
            random.shuffle(self.idx)

        assert self.num_data >= batch_size, \
            "batch_size needs to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    def hard_reset(self):
        """Ignore roll over data and set to start."""
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        if self.iter_next():
            self._prepare_data_label()
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def _prepare_data_label(self):
        """Load data from underlying arrays, internal use only."""
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            l = list(range(self.cursor, self.cursor + self.batch_size))
        else:
            pad = self.batch_size - self.num_data + self.cursor
            l = list(range(self.cursor, self.num_data)) + list(range(0,pad))
        
        offset = int(random.gauss(0, self.setting.sample_num // 8))
        offset = max(offset, -self.setting.sample_num // 4)
        offset = min(offset, self.setting.sample_num // 4)
        sample_num_train = self.setting.sample_num + offset

        l = [self.idx[_] for _ in l] if self.shuffle else l

        data_pad = self.data_pad[l] if isinstance(self.data_pad, np.ndarray) else self.data_pad    
        indices = get_indices(self.batch_size, sample_num_train, data_pad)
        indices_nd = nd.array(indices, dtype=np.int32)

        self.prepare_list = l
        self.indices_nd = indices_nd

    def getdata(self):
        dat = self.data[self.prepare_list,:,:self.setting.data_dim]

        points_sampled = nd.gather_nd(array(dat), indices=self.indices_nd)

        xforms_np, rotations_np = get_xforms(self.batch_size, scaling_range=self.setting.scaling_range
            , rotation_range=self.setting.rotation_range)
        points_xformed = nd.batch_dot(points_sampled, nd.array(xforms_np))
        points_augmented = augment(points_xformed, nd.array(xforms_np), self.setting.jitter)
        return [points_augmented]

    def getlabel(self):
        if len(self.label.shape) == 2:
            label = self.label[self.prepare_list,:]
            labels_sampled = nd.gather_nd(array(label), indices=self.indices_nd)
            return [labels_sampled]
        else:
            label = self.label[self.prepare_list]
            return [label]

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0