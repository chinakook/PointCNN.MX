# coding: utf-8

import mxnet as mx

@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)

def get_shape(x):
    if isinstance(x, mx.nd.NDArray):
        return x.shape
    elif isinstance(x, mx.symbol.Symbol):
        _,x_shape,_=x.infer_shape_partial()
        return x_shape[0] 