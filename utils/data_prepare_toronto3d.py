from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_ply import read_ply
from helper_tool import DataProcessing as DP

anno_paths = glob.glob(os.path.join('/media/cug210/data/ZZY/toronto_unprocess/new_toronto_3d', '*.ply'))

gt_class = [x.rstrip() for x in open('/media/cug210/data/ZZY/toronto_unprocess/new_toronto_3d/toronto_3d_classes_8.txt')]
gt_class2label = {cls: i for i, cls in enumerate(gt_class)}

sub_grid_size = 0.06
original_pc_folder = join('/media/cug210/data/ZZY/toronto_process', 'original_ply')
sub_pc_folder = join('/media/cug210/data/ZZY/toronto_process', 'input_{:.3f}'.format(sub_grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
out_format = '.ply'


def convert_pc2ply(anno_path, save_path):
    """
    Convert original dataset files to ply file (each line is XYZRGBL).
    We aggregated all the points from each instance in the room.
    :param anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
    :param save_path: path to save original point clouds (each line is XYZRGBL)
    :return: None
    """
    # data_list = []
    #
    # for f in glob.glob(join(anno_path, '*.txt')):
    #     class_name = os.path.basename(f).split('_')[0]
    #     if class_name not in gt_class:  # note: in some room there is 'staris' class..
    #         class_name = 'clutter'
    #     pc = pd.read_csv(f, header=None, delim_whitespace=True).values
    #     labels = np.ones((pc.shape[0], 1)) * gt_class2label[class_name]
    #     data_list.append(np.concatenate([pc, labels], 1))  # Nx7
    #
    # pc_label = np.concatenate(data_list, 0)
    # xyz_min = np.amin(pc_label, axis=0)[0:3]
    # pc_label[:, 0:3] -= xyz_min

    process_data = read_ply(anno_path)
    xyz = np.vstack((process_data['x'], process_data['y'], process_data['z'])).T.astype(np.float32)
    colors = np.vstack((process_data['red'], process_data['green'], process_data['blue'])).T.astype(np.uint8)
    labels = np.expand_dims(process_data['label'], axis=-1).astype(np.uint8)
    write_ply(save_path, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    # save sub_cloud and KDTree file
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = join(sub_pc_folder, save_path.split('/')[-1][:-4] + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz)
    kd_tree_file = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = join(sub_pc_folder, str(save_path.split('/')[-1][:-4]) + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)


if __name__ == '__main__':
    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for annotation_path in anno_paths:
        print(annotation_path)
        elements = str(annotation_path).split('/')
        out_file_name = elements[-1]
        print(out_file_name)
        convert_pc2ply(annotation_path, join(original_pc_folder, out_file_name))
