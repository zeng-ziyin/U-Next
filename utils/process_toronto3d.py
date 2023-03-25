import numpy as np
import glob, os

from helper_ply import read_ply
from helper_ply import write_ply

data_dir = r'G:\dl_data\Toronto_3D'
save_dir = r'G:\dl_data\new_toronto_3d'


result_path = glob.glob(os.path.join(data_dir, '*.ply'))
result_path = np.sort(result_path)

for file_name in result_path:
    del_idx = []
    orig_data = read_ply(file_name)
    points = np.vstack((orig_data['x'], orig_data['y'], orig_data['z'], orig_data['red'], orig_data['green'],
                        orig_data['blue'], orig_data['scalar_Label'])).T
    points[:, 0] -= 627285
    points[:, 1] -= 4841948
    for i in range(points.shape[0]):
        if points[i, 6] == 0:
            del_idx.append(i)
            pass
    points = np.delete(points, del_idx, axis=0)
    points[:, 6] -= 1
    save_path = os.path.join(save_dir, file_name.split('\\')[-1][:-4] + '.ply')
    write_ply(save_path, points, ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
