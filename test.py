import torch
from utils.utils import RoboOccDataset,color_map,post_process,gt_occ_lable_to_point_cloud
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import time
import cv2

if __name__=="__main__":
    #open3d display setting
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='occ', width=1200, height=900)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])#backgroud color
    opt.point_size = 4
    pred_occ_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pred_occ_pcd)
    to_reset = True

    dataset_dir = './lg_train_isaaclab_infos.pkl'
    test_dataset = RoboOccDataset(dataset_dir)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1, prefetch_factor=1)
    
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    RoboOcc = torch.jit.load("roboocc_scripted.pt",map_location="cuda:0")
    RoboOcc.eval()

    for sample_idx, data in enumerate(test_dataloader):
        img = data['img'].squeeze(0).to(device)
        gt_occ = data['gt_occ'].squeeze(0).to(device)

        with torch.no_grad():
            pred_occ = RoboOcc(img)#inferrence

            ###########################################
            #post process and display and save
            ###########################################
            pred_occ_array = post_process(pred_occ)

            pred_occ_pcd.points = o3d.utility.Vector3dVector(pred_occ_array[:, :3])
            #rotate 180 degrees counterclockwise about the X-axis, just for display
            R = pred_occ_pcd.get_rotation_matrix_from_xyz( (np.pi, 0, 0))
            pred_occ_pcd.rotate(R,center=(0, 0, 0))
            #color rendering
            pred_labels = pred_occ_array[:, 3].astype(int)
            color = color_map[pred_labels] / 255.0
            pred_occ_pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
            #pcd save
            pred_file_path = 'pred/' + str(sample_idx) + '_pred_occ.ply'
            o3d.io.write_point_cloud(pred_file_path, pred_occ_pcd) #save
            
            #################################
            #image read and display and save
            #################################
            img_source = cv2.imread(data['cams_file_path'][0])
            img_source = cv2.resize(img_source,(616,344))
            image_rgb = cv2.cvtColor(img_source,cv2.COLOR_BGR2RGB)
            img_save_path = 'pred/' + str(sample_idx) + '_img.jpg'
            cv2.imwrite(img_save_path,img_source)
            cv2.imshow('img_source',img_source)

            #pred occ display
            if to_reset:
                vis.reset_view_point(True)
                to_reset = False
            vis.update_geometry(pred_occ_pcd)
            vis.poll_events()
            vis.update_renderer()

            #########################
            #occ gt save for compare
            #########################
            occ_gt_label = data['gt_occ'].squeeze(0)
            gt_occ_array = gt_occ_lable_to_point_cloud(occ_gt_label)
            gt_occ_pcd = o3d.geometry.PointCloud()
            gt_occ_pcd.points = o3d.utility.Vector3dVector(gt_occ_array)
            #rotate 180 degrees counterclockwise about the X-axis, for consistent 
            R = gt_occ_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
            gt_occ_pcd.rotate(R,center=(0, 0, 0))
            #color rendering
            occ_gt_label = occ_gt_label.numpy()
            occ_gt_label = occ_gt_label[occ_gt_label != 0]
            occ_gt_label = occ_gt_label.astype(int)
            color = color_map[occ_gt_label] / 255.0
            gt_occ_pcd.colors = o3d.utility.Vector3dVector(color[..., :3])
            #pcd save
            pred_file_path = 'pred/' + str(sample_idx) + '_gt_occ.ply'
            o3d.io.write_point_cloud(pred_file_path, gt_occ_pcd) #save

            print('sample {} is ok..'.format(sample_idx))

            time.sleep(1)


    print('ok!')