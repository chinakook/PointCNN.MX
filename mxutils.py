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

class Weightsoftmax(mx.operator.CustomOp):
    def __init__(self):
        super(Weightsoftmax, self).__init__()
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        # y = np.exp(x - x.max(axis=1))
        # y /= y.sum(axis=1)
        
        t = x.max(axis=-1)
        t = t[:,:,np.newaxis]
        
        y = np.exp(x - t)
        v =y.sum(axis=-1)
        v = v[:,:,np.newaxis]
        y = y/v
        #print("Y", y.shape)
        #label = np.concatenate((b_label,f_label),axis=1)      
        #print "foward"
        self.assign(out_data[0], req[0], mx.nd.array(y))
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        prob = out_data[0].asnumpy()
        #weight = np.array([0.096295,1.411320])
        #weight = np.array([0.01,1.])
        weight = np.array([0.04 ,1])
        
        f_label = in_data[1].asnumpy()
        f_label = f_label.astype(np.float32)
        f_label = f_label[:,:,np.newaxis]
        label = np.concatenate((1-f_label,f_label),axis=2)
    
        out = prob - label

        out[:,:,0] *= weight[0]
        out[:,:,1] *= weight[1]
        #print("L", out.shape)
        out/=(out.shape[0]*out.shape[1])
        #print out[0,:,0:10,0:10]
        #m = raw_input()
        #print "backward"
        self.assign(in_grad[0], req[0], mx.nd.array(out))
@mx.operator.register("weightsoftmax")
class WeightsoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(WeightsoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['indata', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
         
        data_shape = in_shapes[0]
        # weight_shape = in_shapes[1]
        label_shape = in_shapes[1]

        output_shape = data_shape
        
        #print data_shape,weight_shape,label_shape
        return [data_shape,label_shape], [output_shape],[]


    def create_operator(self, ctx, shapes, dtypes):
        return Weightsoftmax()  
