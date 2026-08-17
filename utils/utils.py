import numpy as np
from numpy import random
import torch
import os
# import h5py
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data
import pickle
import cv2
from torchvision import transforms
from PIL import Image
import open3d as o3d

class RoboOccDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_dir):
        super().__init__()
        self.dir = dataset_dir
        # 打开一个文件用于写入
        with open(dataset_dir, 'rb') as f:
            try:
                self.data_info = pickle.load(f)
                print('ok')
            except EOFError:
                return None
    
    def __len__(self):
        return len(self.data_info['infos'])
    
    def __getitem__(self, index):
        data_item_info = self.data_info['infos'][index]
        cam_file = data_item_info['camera']
        occ_gt_file = data_item_info['occ_gt']

        result = dict()
     
        #image process
        img = cv2.imread(cam_file).astype(np.float32)
        img = cv2.resize(img,(480,256),interpolation=cv2.INTER_LINEAR )
        img = np.expand_dims(img, axis=0)
        img = img.transpose(0,3,2,1)#!opencv
        result['img'] = img

        #occ gt process
        occ = np.load(occ_gt_file)
        occ = occ['semantic'].astype(np.float32)
        gt_occ = np.zeros([120, 60, 120])
        gt_occ[:120,:40,:120] = occ
        result['gt_occ'] = gt_occ

        result['cams_file_path']=cam_file
        result['gt_file_path']=occ_gt_file

        return result

color_map = np.array(
                [   [0, 0, 0, 255],  # unoccupancy
                    [160, 32, 240, 255],  #  0
                    [135, 60, 0, 255],  #  1
                    [255, 255, 0, 255],  #  2
                    [0, 255, 255, 255],  #  3
                    [255, 192, 203, 255],  #  4
                    [200, 180, 0, 255],  #  5
                    [255, 0, 0, 255],  #  6
                    [255, 240, 150, 255],  #  7
                    [255, 120, 50, 255],  #  8
                    [255, 0, 255, 255],  # 9
                    [139, 137, 137, 255],  #  10
                    [75, 0, 75, 255],  #  11
                    [150, 240, 80, 255],  # 12 
                    [230, 230, 250, 255], # 13
                    [0, 175, 0, 255],  # 14
                    [0, 150, 245, 255],  #  15
                    [160, 32, 240, 255],  #  16
                    [135, 60, 0, 255],  #  17
                    [255, 255, 0, 255],  #  18
                    [0, 255, 255, 255],  #  19
                    [255, 192, 203, 255],  #  20
                    [200, 180, 0, 255],  #  21
                    [255, 0, 0, 255],  #  22
                    [160, 32, 240, 255],  #  23
                    [135, 60, 0, 255],  #  24
                    [255, 255, 0, 255],  #  25
                    [0, 255, 255, 255],  #  26
                ]
            )

def post_process(pred_occ):
    filter_1 = torch.sigmoid(pred_occ['binary_pred'][:,0,...])#!
    class_num = pred_occ['shape_and_semantic_pred'][0].shape[1]

    conf1 = 0.6
    conf2 = 0.6
    for i in range(pred_occ['binary_pred'].shape[0]):
        filter_1 = filter_1 > conf1

        occ_before_upsample_reshape = []
        for n in range(27):
            occ_before_upsample_reshape.append( torch.zeros([1,class_num,40, 20, 40]).to(torch.device(pred_occ['binary_pred'].device)).type(torch.float) .permute(0,2,3,4,1))

        pred = pred_occ['shape_and_semantic_pred']
        for n in range(27):
            pred_item = pred[n]
            pred_item = pred_item.permute(0,2,3,4,1)
            occ_before_upsample_reshape [n][filter_1] = pred_item[filter_1]
            occ_before_upsample_reshape[n] =occ_before_upsample_reshape[n].permute(0,4,1,2,3)

        occ_size = [120,60,120]
        occ_result = torch.zeros([1,class_num, occ_size[0], occ_size[1], occ_size[2]]).to(pred[0].device).type(torch.float) #!
            #----------------------------111----------------------------------#
        occ_result[:,:,0:occ_size[0]:3,0:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[0]
        occ_result[:,:,1:occ_size[0]:3,0:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[1]
        occ_result[:,:,2:occ_size[0]:3,0:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[2]

        occ_result[:,:,0:occ_size[0]:3,1:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[3]
        occ_result[:,:,1:occ_size[0]:3,1:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[4]
        occ_result[:,:,2:occ_size[0]:3,1:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[5]

        occ_result[:,:,0:occ_size[0]:3,2:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[6]
        occ_result[:,:,1:occ_size[0]:3,2:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[7]
        occ_result[:,:,2:occ_size[0]:3,2:occ_size[1]:3,0:occ_size[2]:3]=occ_before_upsample_reshape[8]
        #----------------------------222----------------------------------#
        occ_result[:,:,0:occ_size[0]:3,0:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[9]
        occ_result[:,:,1:occ_size[0]:3,0:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[10]
        occ_result[:,:,2:occ_size[0]:3,0:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[11]

        occ_result[:,:,0:occ_size[0]:3,1:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[12]
        occ_result[:,:,1:occ_size[0]:3,1:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[13]
        occ_result[:,:,2:occ_size[0]:3,1:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[14]

        occ_result[:,:,0:occ_size[0]:3,2:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[15]
        occ_result[:,:,1:occ_size[0]:3,2:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[16]
        occ_result[:,:,2:occ_size[0]:3,2:occ_size[1]:3,1:occ_size[2]:3]=occ_before_upsample_reshape[17]
        #----------------------------333----------------------------------#
        occ_result[:,:,0:occ_size[0]:3,0:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[18]
        occ_result[:,:,1:occ_size[0]:3,0:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[19]
        occ_result[:,:,2:occ_size[0]:3,0:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[20]

        occ_result[:,:,0:occ_size[0]:3,1:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[21]
        occ_result[:,:,1:occ_size[0]:3,1:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[22]
        occ_result[:,:,2:occ_size[0]:3,1:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[23]

        occ_result[:,:,0:occ_size[0]:3,2:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[24]
        occ_result[:,:,1:occ_size[0]:3,2:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[25]
        occ_result[:,:,2:occ_size[0]:3,2:occ_size[1]:3,2:occ_size[2]:3]=occ_before_upsample_reshape[26]
        
        tensor_2_equal = torch.zeros([1,class_num, 1,1,1]).to(pred[0].device).type(torch.float) #!
        occ_result_filter = torch.eq(occ_result,tensor_2_equal)
        occ_result_filter = ~occ_result_filter
        occ_result_filter = torch.sum(occ_result_filter,dim=1,keepdim=False)
        occ_result_filter = occ_result_filter >0

        occ_result = occ_result.permute(0,2,3,4,1)

        x = torch.linspace(0, 119, occ_size[0])
        y = torch.linspace(0, 59, occ_size[1])
        z = torch.linspace(0, 119, occ_size[2])
        X, Y, Z = torch.meshgrid(x, y, z)
        vv = torch.stack([X, Y, Z], dim=-1).to(occ_result.device).unsqueeze(0)
    
        occ_result = occ_result[occ_result_filter]
        occ_result_binary = occ_result[:,0]
        occ_result_semantic = occ_result[:,1:]

        occ_result_binary = occ_result_binary.sigmoid()
        filter_2 = occ_result_binary < (1-conf2)
        occ_result_semantic = occ_result_semantic[filter_2]

        conf , j = occ_result_semantic.max(1,keepdim=True)
        j += 1
        vertices = vv[occ_result_filter]
        vertices = vertices[filter_2]
        vertices = torch.cat([vertices,j],dim=1)

        occ_range = [-30.0,-16.0,0.2,30.0,4.0,60.0]
        vertices[:, 0] = (vertices[:, 0] + 0.5) * (occ_range[3] - occ_range[0]) / occ_size[0] + occ_range[0]
        vertices[:, 1] = (vertices[:, 1] + 0.5) * (occ_range[4] - occ_range[1]) / occ_size[1] + occ_range[1]
        vertices[:, 2] = (vertices[:, 2] + 0.5) * (occ_range[5] - occ_range[2]) / occ_size[2] + occ_range[2]
        vertices = vertices.cpu().numpy()

    return vertices

def gt_occ_lable_to_point_cloud(occ_label):
    x = torch.linspace(0, 119, 120)
    y = torch.linspace(0, 59, 60)
    z = torch.linspace(0, 119, 120)
    X, Y, Z = torch.meshgrid(x, y, z)
    gt_occ = torch.stack([X, Y, Z], dim=-1).to(occ_label.device)
    gt_occ = gt_occ[occ_label!=0]
    gt_occ = gt_occ.cpu().numpy()
    occ_range = [-30.0,-16.0,0.2,30.0,4.0,60.0]
    occ_size = [120,60,120]
    gt_occ[:, 0] = (gt_occ[:, 0] + 0.5) * (occ_range[3] - occ_range[0]) /  occ_size[0]  + occ_range[0]
    gt_occ[:, 1] = (gt_occ[:, 1] + 0.5) * (occ_range[4] - occ_range[1]) /  occ_size[1]  + occ_range[1]
    gt_occ[:, 2] = (gt_occ[:, 2] + 0.5) * (occ_range[5] - occ_range[2]) /  occ_size[2]  + occ_range[2]

    return gt_occ



