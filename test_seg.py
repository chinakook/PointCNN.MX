import os
import sys
import math
from laspy import file as File
from laspy import header
from laspy import util
from transforms3d.euler import euler2mat
import numpy as np
import copy
import os
import time
import sys
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT']='0'
import argparse
import math
import random
import time
import logging
logging.basicConfig(level=logging.INFO)

import mxnet as mx
from mxnet import nd
import mxnet.gluon as gluon
from mxutils import get_shape

from pointcnn import PointCNN, custom_metric
from sampleiter import get_indices, get_xforms, augment
from dotdict import DotDict
import h5py
import collections
import data_utils
#from test_batch import las2npy,npy2h5,save_las
import shutil
from datetime import datetime
"""Testing On Segmentation Task."""
'''
    input: las file
    output: las file with pred
    ########################################################################################################
    python3 test_split.py -l /mnt/15F1B72E1A7798FD/DK2/cuda-workspace/VeloCompositor/Release/part_pred \
            -k ./model/improve/pointcnn_seg_road1_5899_2018-03-30-11-36-33/ckpts/iter-1900 -m pointcnn_seg -x road1
    ########################################################################################################
'''


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

def las2npy(p_path,out_filename):
    assert(p_path.endswith(".las"))
    x_flag = 1
    y_flag = 1
    if (p_path.endswith("1_rect.las")):        
        y_flag = -1
    if (p_path.endswith("0_rect.las")):
        x_flag = -1
        y_flag = -1
    if (p_path.endswith("2_rect.las")):
        x_flag = -1    
    road_file = File.File(p_path)
    point_records = road_file.points
    tmp_np = np.zeros(shape=(point_records.shape[0],3))
    x_dimension = road_file.X
    y_dimension = road_file.Y
    z_dimension = road_file.Z
    scale = road_file.header.scale[0]
    #ints = road_file.intensity
    clas = road_file.raw_classification
    scale_x = x_dimension*scale*x_flag
    scale_y = y_dimension*scale*y_flag
    scale_z = z_dimension*scale 
    tmp_np[:,0] = scale_x
    tmp_np[:,1] = scale_y
    tmp_np[:,2] = scale_z
    # xyz_min = np.amin(tmp_np, axis=0)[0:3]
    xyz_min = [ 6.99960784,  3.1982549 ,  1.12736275]    
    tmp_np[:, 0:3] -= xyz_min
    np.save(out_filename, tmp_np)  

def npy2h5(data_label_folder,h5_save_dir):
    max_point_num = 0    
    for filename in sorted(os.listdir(data_label_folder)):                         
        data_filepath = os.path.join(data_label_folder, filename) 
        
        assert data_filepath.endswith(".npy")        
        coordinates = np.load(data_filepath)
        max_point_num = max(max_point_num, len(coordinates))

    h5_save_dir = h5_save_dir+"/"   
    batch_size = 2048

    data = np.zeros((batch_size, max_point_num, 3))
    data_num = np.zeros((batch_size), dtype=np.int32)
    label = np.zeros((batch_size), dtype=np.int32)
    label_seg = np.zeros((batch_size, max_point_num), dtype=np.int32)



    file_num = 0
    for category in sorted(os.listdir(data_label_folder)):
        data_category_folder = os.path.join(data_label_folder, category)        
        if not os.path.isfile(data_category_folder):
                continue
        file_num = file_num + 1
    idx_h5 = 0
    idx = 0
    f_name = h5_save_dir+"test"
    #save_path = '%s/%s' % (os.path.dirname(data_label_folder), os.path.basename(data_label_folder)[0:-5])        
    filename_txt = '%s_files.txt' % (f_name)

    with open(filename_txt, 'w') as filelist:
        for category in sorted(os.listdir(data_label_folder)):
            coordinates = []
            data_filepath = os.path.join(data_label_folder, category)            
            if not os.path.isfile(data_filepath):
                continue
            
            xyz_rgb_l = np.load(data_filepath)

            # shuffle  
            #np.random.shuffle(xyz_rgb_l)
            
            coordinates = xyz_rgb_l[:,0:3]                
            idx_in_batch = idx % batch_size
            # data[idx_in_batch, len(corredinates),3]
            data[idx_in_batch, 0:len(coordinates), ...] = coordinates
            # data_num
            data_num[idx_in_batch] = len(coordinates)
            # label 
            label[idx_in_batch] = 0
            
            label_seg[idx_in_batch, 0:len(coordinates)] = 0               
            # print(idx)
            if ((idx + 1) % batch_size == 0) or idx == file_num - 1:
                item_num = idx_in_batch + 1
                filename_h5 = '%s_%d.h5' % (h5_save_dir, idx_h5)
                print('{}-Saving {}...'.format(datetime.now(), filename_h5))
                # save_h5_path = h5_save_dir+os.path.basename(data_label_folder)
                #save_h5_path = h5_save_dir

                # save .h5 filelist
                filelist.write('%s_%d.h5\n' % (h5_save_dir, idx_h5))

                # save h5 file, data data_num label label_seg
                file = h5py.File(filename_h5, 'w')

                file.create_dataset('data', data=data[0:item_num, ...])
                file.create_dataset('data_num', data=data_num[0:item_num, ...])
                file.create_dataset('label', data=label[0:item_num, ...])
                file.create_dataset('label_seg', data=label_seg[0:item_num, ...])
                file.close()                    
                idx_h5 = idx_h5 + 1
            idx = idx + 1

def save_las(inFilename,outFilename):
    road_file = File.File(inFilename)    
    new_header = copy.copy(road_file.header)
    tmp_File = File.File(outFilename, mode = "w",header=new_header)
    for spec in road_file.reader.point_format:
        # print("Copying dimension: " + spec.name)
        in_spec = road_file.reader.get_dimension(spec.name)
        try:
            tmp_File.writer.set_dimension(spec.name, in_spec)
        except(util.LaspyException):
            print("Couldn't set dimension: " + spec.name +
                    " with file format " + str(tmp_File.header.version) +
                    ", and point_format " + str(tmp_File.header.data_format_id))
    road_file.close()
    return tmp_File

def detect(data, valid_num, mod):


    sample_num = 4096
    batch_size = int(math.ceil(data.shape[1]*1.0/sample_num))

    points_batch = data[[0]*batch_size, ...]    

    indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, setting.sample_num, 1))

    point_num = valid_num
    tile_num = int(math.ceil((sample_num * batch_size*1.0) / point_num))

    indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
    np.random.shuffle(indices_shuffle)
    indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
    indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

    data_f = nd.array(points_batch)
    indices_f = nd.array(indices_batch)
    indices_f = nd.transpose(indices_f,(2,0,1))

    data_in = nd.gather_nd(data_f,indices= indices_f)

    seg_res = np.zeros(shape=(batch_size,sample_num,setting.num_class))
    for j in range(batch_size):
        d_in = nd.expand_dims(data_in[j,:,:], 0)
        nb = mx.io.DataBatch(data=[d_in], label=None, pad=None, index=None)
        mod.forward(nb, is_train = False)
        seg_res[j,:,:] = mod.get_outputs()[0].asnumpy()

    probs_2d = np.reshape(seg_res, (sample_num * batch_size, -1))
    predictions = [(-1, 0.0)] * point_num
    for idx in range(sample_num * batch_size):
        point_idx = indices_shuffle[idx]
        point_probs = probs_2d[idx, :]
        prob = np.amax(point_probs)
        seg_idx = np.argmax(point_probs)
        if prob > predictions[point_idx][1]:
            predictions[point_idx] = (seg_idx, prob)

    labels = []    
    for seg_idx, _ in predictions:
        labels.append(seg_idx)
    return labels

def process(infile,outfile,resfle):
    # save rect and pred file

    save_rect = True
    
    l = File.File(infile,mode="r")
   
    m = l.header.get_softwareid()   
    k = float(m[:m.find('\x00')])
    offset = l.header.offset
    order='rxyz'
    rotation = euler2mat(0, 0, k, order)
    tmp = np.zeros(shape=(l.X.shape[0],3))
    tmp[:,0] = l.X
    tmp[:,1] = l.Y
    tmp[:,2] = l.Z
    tmp = tmp * l.header.scale
    af = np.dot(tmp,rotation)
    # bound = ((af[:,0]>-30) & (af[:,0]<30) & (af[:,1]>-10) & (af[:,1]<10) & (af[:,2]>-2.0) & (af[:,2]<5)
    #      & ((af[:,0] < -1.5)  | (af[:,0] > 0.5) | (af[:,1] < -2.5) | (af[:,1] > -0.5) | (af[:,2] < -0.5) | (af[:,2] > 0.5)))

    # print af.shape[0],bound.sum()
    # keep_points = bound
    
    new_header = copy.copy(l.header)
    tmp_las = outfile
    w_las = resfle
    
    w =  File.File(w_las,mode = "w",header=new_header)
    for spec in l.reader.point_format:
    # print("Copying dimension: " + spec.name)
        in_spec = l.reader.get_dimension(spec.name)
        try:
            w.writer.set_dimension(spec.name, in_spec)
        except(util.LaspyException):
            print("Couldn't set dimension: " + spec.name +
                    " with file format " + str(w.header.version) +
                    ", and point_format " + str(w.header.data_format_id))
    w.close()
    #################################### save_rect #################################################
    if(save_rect):
        n = File.File(tmp_las,mode = "w",header=new_header)
        points_kept = l.points
        n.points = points_kept

        # n.X = af[:,0] / l.header.scale[0] + offset[0] 
        # n.Y = af[:,1] / l.header.scale[1] + offset[1] 
        # n.Z = af[:,2] / l.header.scale[2] + offset[2]
        
        n.X = af[:,0] / l.header.scale[0] 
        n.Y = af[:,1] / l.header.scale[1]
        n.Z = af[:,2] / l.header.scale[2]

        n.close()  

def split_las(or_dir,save_dir):
    for num,j in enumerate(os.listdir(or_dir)):
        p_0 =[]
        p_1 = []
        p_2 = []
        p_3 = []
        or_las = or_dir + "/" + j
        o_f = File.File(or_las)
        new_header = copy.copy(o_f.header)
        p1 = np.zeros(shape=(o_f.X.shape[0],3))
        p1[:,0] = o_f.X * o_f.header.scale[0]
        p1[:,1] = o_f.Y * o_f.header.scale[0]
        p1[:,2] = o_f.Z * o_f.header.scale[0]

        t = []
        for i in range(p1.shape[0]):
            if(p1[i,1]<=-1.8):
                if(p1[i,0]<=0):
                    p_0.append(i)
                else:
                    p_1.append(i)

            else:        
                if(p1[i,0]<=0):
                    p_2.append(i)
                else:
                    p_3.append(i)
        t.append(p_0)
        t.append(p_1)
        t.append(p_2)
        t.append(p_3)
        # print(p1[:,1])
        for x in range(4):
            n_path = save_dir+"/"+j[:-4]+"_"+str(x)+".las"
            n_rect_path = save_dir+"/"+j[:-4]+"_"+str(x)+"_rect"+".las"
            n_savepath = save_dir+"/"+j[:-4]+"_"+str(x)+"_pred"+".las"
            n = File.File(n_path,mode = "w",header=new_header)
            keep_points = np.array(t[x])
            points_kept = o_f.points[keep_points]            
            n.points = points_kept
            n.close()  
            process(n_path,n_rect_path,n_savepath)         

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--laspath', '-l', help='Path to input las file (.las)', required=True)

    las_dir = "/mnt/15F1B72E1A7798FD/part" #args.laspath
    npy_dir = os.path.dirname(las_dir)+"/"+os.path.basename(las_dir)+"_npy"
    h5_dir = os.path.dirname(las_dir)+"/"+os.path.basename(las_dir)+"_h5"
    pre_dir = os.path.dirname(las_dir)+"/"+os.path.basename(las_dir)+"_pred"#pred_test"
    split_flag = False
    if not os.path.exists(npy_dir):
        os.mkdir(npy_dir)
    if not os.path.exists(h5_dir):
        os.mkdir(h5_dir)
    if not os.path.exists(pre_dir):
        os.mkdir(pre_dir)

    output_filelist = []
    input_filelist = []
    
    if split_flag:
        split_las(las_dir,pre_dir)
        for i in sorted(os.listdir(pre_dir)):
            if i.endswith("_rect.las"):
                laspath = os.path.join(pre_dir,i)
                input_filelist.append(laspath)
                npypath = os.path.join(npy_dir,i[:-3]+"npy")
                output_filelist.append(os.path.join(pre_dir, i[:-8]+'pred.las'))
                las2npy(laspath,npypath)       
    else:
        for i in sorted(os.listdir(las_dir)):
            if(i.endswith(".las")):
                laspath = os.path.join(las_dir,i)
                npypath = os.path.join(npy_dir,i[:-3]+"npy")
                rectpath = os.path.join(pre_dir,i[:-4]+"_rect.las")
                predpath = os.path.join(pre_dir, i[:-4]+'_pred.las')
                input_filelist.append(rectpath)
                output_filelist.append(predpath)
                #process(laspath,rectpath,predpath)
                # las2npy(rectpath,npypath)

    # npy2h5(npy_dir,h5_dir)
    # sys.exit(0)

    #Leng = 1
    ctx = [mx.gpu(1)]

    sym, arg_params, aux_params = mx.model.load_checkpoint('/home/dingkou/dev/PointCNN.MX/p_seg',401)
    all_layers = sym[0].get_internals()
    new_sym = mx.sym.softmax(all_layers['fc_dense5_fwd_output'], axis=-1)

    mod = mx.mod.Module(symbol=new_sym, context=ctx, data_names=['data'], label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (2, 4096, 3))], label_shapes=None)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)

    net = PointCNN(setting, 'segmentation', with_feature=False, prefix="")
    net.hybridize()
    var = mx.sym.var('data', shape=(1, 4096, 3))
    probs = net(var)
    mod._symbol = mx.sym.softmax(probs, axis=-1)
    mod.binded=False

    mod.bind(for_training=False, data_shapes=[('data', (1, 4096, 3))]
                , label_shapes=None, shared_module=mod
            )
    ###################################################################################

    npy_txt = h5_dir+"/test_files.txt"

    data, _, data_num, _ = data_utils.load_seg(npy_txt)

    batch_num = data.shape[0]

    for i in range(batch_num):
        w =  File.File(output_filelist[i],mode = "rw")
        in_data = data[i]
        in_data = in_data[np.newaxis,:]
        valid_num = data_num[i]
        label = detect(in_data, valid_num, mod)
        w.raw_classification = label
        w.close()




if __name__ == '__main__':
    main()
