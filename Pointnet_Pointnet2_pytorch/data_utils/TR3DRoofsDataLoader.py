# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
import laspy
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

# Colors made with: https://mokole.com/palette.html
# A tool to generate any number of visually distinct colors
roof_plane_to_color = {0: '#A6CEE3', 1: '#1F78B4', 2: '#B2DF8A', 3: '#33A02C', 4: '#FB9A99',
                       5: '#E31A1C', 6: '#FDBF6F', 7: '#FF7F00', 8: '#CAB2D6', 9: '#6A3D9A', 10: '#FFFF99', 11: '#B15928'}
sem_label_to_inst = {0: [0, 1], 1: [2, 3],
                     2: [4, 5], 3: [6, 7], 4: [8, 9, 10, 11]}


def hex_to_rgb(hex): return tuple(int(hex[i:i+2], 16) for i in (1, 3, 5))


class TR3DRoofsDataset(Dataset):
    # inst_type: 'sem' | 'inst'
    def __init__(self, root='./data/TR3DRoofs', npoints=1024, split='train', seg_type='inst'):
        self.npoints = npoints
        self.root = root
        self.seg_type = seg_type

        # Read the data
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_viz_file_list.json'), 'r') as f:
            viz_ids = set([str(d) for d in json.load(f)])

        data_dir = os.path.join(self.root, '00000000')
        self.fns = sorted(os.listdir(data_dir))  # file names

        # fn[0:-4] is file extention '.txt' so remove it
        if split == 'trainval':
            self.fns = [fn for fn in self.fns if (
                (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
        elif split == 'train':
            self.fns = [fn for fn in self.fns if fn[0:-4] in train_ids]
        elif split == 'val':
            self.fns = [fn for fn in self.fns if fn[0:-4] in val_ids]
        elif split == 'test':
            self.fns = [fn for fn in self.fns if fn[0:-4] in test_ids]
        elif split == 'viz':
            self.fns = [fn for fn in self.fns if fn[0:-4] in viz_ids]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        # Add dir path to file names
        self.fns = [os.path.join(data_dir, fn) for fn in self.fns]

        # Store some points in a cache
        self.cache = {}  # from index to (point_set, roof_type, seg) tuple
        self.cache_size = 20000

    def store_segmented_roof(self, index, point_set, seg_values, file_path):
        # Read roof_type
        data = np.loadtxt(self.fns[index]).astype(np.float32)
        roof_id_to_type = {1: 'Flat', 2: 'Hipped', 3: 'Gabled',
                           4: 'Corner Element', 5: 'T-Element', 6: 'Cross Element', 7: 'Combination'}
        roof_id = data[:, 3].astype(np.int32)[0]
        roof_type = roof_id_to_type[roof_id]

        # Read points
        xyz = np.ascontiguousarray(point_set[0], dtype='float32')

        # For each seg_value, return correct color
        if self.seg_type == 'inst':
            def colorize(x): return hex_to_rgb(roof_plane_to_color[x])
        else:
            def colorize(x): return hex_to_rgb(
                roof_plane_to_color[sem_label_to_inst[x][0]])

        rgb = np.ascontiguousarray([colorize(s)
                                   for s in seg_values], dtype='uint8')

        # if roof_id == 7: # Combination
        #     print(seg_values[120])
        #     print(rgb[120])

        # Set values
        header = laspy.LasHeader(version='1.4', point_format=7)
        las = laspy.LasData(header)
        las.x = xyz[0, :]
        las.y = xyz[1, :]
        las.z = xyz[2, :]
        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]
        # las.Intensity = i
        las.classification = seg_values

        # Store the file
        file_name = os.path.join(file_path, roof_type + '.las')
        las.write(file_name)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            # Data structure
            # x, y, z, roof type, instance label, semantic label
            # xyz is normalized for the whole roof
            data = np.loadtxt(self.fns[index]).astype(np.float32)
            point_set = data[:, 0:3]  # already normalized

            cls = np.array([0]).astype(np.int32)

            # Read every roof_type
            # roof_type = data[:, 3].astype(np.int32)

            # Every point int he same file has the same roof_type, so convert it to ine singel array with one element
            # OBS: Since the roof_types are 1-indexed in the dataset we need to 0-index it here
            # roof_type = np.array([roof_type[0] - 1]).astype(np.int32)

            if self.seg_type == "inst":
                seg = data[:, 4].astype(np.int32)
            else:
                seg = data[:, 5].astype(np.int32)

            # TODO: only store the last cache_size points
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        # Resample
        '''
        # I do not include this, as the network will not learn double stacked points on the large engough samples
        if len(point_set) >= self.npoints:
            # Do not allow double points
            choice = np.random.choice(len(point_set), self.npoints, replace=False)
        else:
            # Allow if not enough points
            choice = np.random.choice(len(point_set), self.npoints, replace=True)
        '''
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.fns)


def visualize_gt(data_root):
    with open(os.path.join(data_root, 'train_test_split', 'shuffled_viz_file_list.json'), 'r') as f:
        viz_ids = set([str(d) for d in json.load(f)])

    data_dir = os.path.join(data_root, '00000000')

    fns = sorted(os.listdir(data_dir))  # file names

    current_fns = [fn for fn in fns if fn[0:-4] in viz_ids]

    # Add dir path to file names
    current_fns = [os.path.join(data_dir, fn) for fn in current_fns]

    for fn in current_fns:
        # Read data
        data = np.loadtxt(fn).astype(np.float32)
        roof_id_to_type = {1: 'Flat', 2: 'Hipped', 3: 'Gabled',
                           4: 'Corner Element', 5: 'T-Element', 6: 'Cross Element', 7: 'Combination'}
        roof_id = data[:, 3].astype(np.int32)[0]
        roof_type = roof_id_to_type[roof_id]

        # Read points
        xyz = np.ascontiguousarray(data[:, 0:3], dtype='float32')

        roof_inst_points = np.ascontiguousarray(data[:, 4], dtype='uint8')
        roof_sem_points = np.ascontiguousarray(data[:, 5], dtype='uint8')

        for file_path, seg_values in [('log/part_seg/gt', roof_sem_points), ('log/inst_seg/gt', roof_inst_points)]:
            # For each seg_value, return correct color
            if 'inst' in file_path:
                def colorize(x): return hex_to_rgb(roof_plane_to_color[x])
            else:
                def colorize(x): return hex_to_rgb(
                    roof_plane_to_color[sem_label_to_inst[x][0]])
            rgb = np.ascontiguousarray([colorize(s)
                                       for s in seg_values], dtype='uint8')

            # Set values
            header = laspy.LasHeader(version='1.4', point_format=7)
            las = laspy.LasData(header)

            las.x = xyz[:, 0]
            las.y = xyz[:, 1]
            las.z = xyz[:, 2]
            las.red = rgb[:, 0]
            las.green = rgb[:, 1]
            las.blue = rgb[:, 2]
            # las.Intensity = i
            las.classification = seg_values

            # Store the file
            file_name = os.path.join(file_path, roof_type + '.las')
            las.write(file_name)


# Test implementation
if __name__ == '__main__':
    import torch

    data_root = 'data/tr3d_roof_segmented_dataset/'

    # print('\n\nVisualizing gt...', end='')
    # visualize_gt(data_root)
    # print(' done!')

    # Read the data
    with open(os.path.join(data_root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
        train_ids = set([str(d) for d in json.load(f)])
    with open(os.path.join(data_root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
        val_ids = set([str(d) for d in json.load(f)])
    with open(os.path.join(data_root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
        test_ids = set([str(d) for d in json.load(f)])

    data_dir = os.path.join(data_root, '00000000')
    fns = sorted(os.listdir(data_dir))  # file names

    # fn[0:-4] is file extention '.txt' so remove it

    sem_to_label = {0: 'Flat', 1: 'Hipped', 2: 'Gabled', 3: 'Corner Element',
                    4: 'T-Element', 5: 'Cross Element', 6: 'Combination'}
    class_distribution = {'Flat': 0, 'Hipped': 0, 'Gabled': 0,
                          'Corner Element': 0, 'T-Element': 0, 'Cross Element': 0, 'Combination': 0}

    plane_names = {0: 'Rectangular', 1: 'Isosceles trapezoid',
                   2: 'Triangular', 3: 'Parallelogram', 4: 'Ladder shapes'}

    sum_sem_points = [0 for _ in range(5)]
    sum_sem_planes = [0 for _ in range(5)]

    sum_inst_points = [0 for _ in range(12)]
    sum_inst_planes = [0 for _ in range(12)]

    all_total = 0
    viz_roofs = [[] for _ in range(7)]
    for split in ['train', 'val', 'test']:
        if split == 'trainval':
            current_fns = [fn for fn in fns if (
                (fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
        elif split == 'train':
            current_fns = [fn for fn in fns if fn[0:-4] in train_ids]
        elif split == 'val':
            current_fns = [fn for fn in fns if fn[0:-4] in val_ids]
        elif split == 'test':
            current_fns = [fn for fn in fns if fn[0:-4] in test_ids]

        # Add dir path to file names
        current_fns = [os.path.join(data_dir, fn) for fn in current_fns]

        for fn in current_fns:
            # Read data
            data = np.loadtxt(fn).astype(np.float32)

            # Read every roof_type
            # OBS: Since the roof_types are 1-indexed in the dataset we need to 0-index it here
            roof_type = data[:, 3].astype(np.int32)
            roof_type = int(roof_type[0] - 1)

            # x, y, z, roof type, instance label, semantic label
            # TODO: Check if this shit works
            roof_inst_points = data[:, 4].astype(np.int32)
            roof_sem_points = data[:, 5].astype(np.int32)

            roof_inst_planes = np.unique(roof_inst_points)
            for i in roof_inst_planes:
                sum_inst_points[i] += np.sum(roof_inst_points == i)
                sum_inst_planes[i] += 1

            roof_sem_planes = np.unique(roof_sem_points)
            for i in roof_sem_planes:
                sum_sem_points[i] += np.sum(roof_sem_points == i)
                sum_sem_planes[i] += 1

            if len(viz_roofs[roof_type]) < 3:
                viz_roofs[roof_type].append(str(fn.split('/')[-1]))

            # Update distribution
            class_distribution[sem_to_label[roof_type]] += 1

        # Class
        print('Class distribution in split:', split)
        total = 0
        for cat in sorted(class_distribution.keys()):
            print('number of instances in class %s %f' %
                  (cat + ' ' * (20 - len(cat)), class_distribution[cat]))
            total += class_distribution[cat]
        all_total += total
        print('Total number of instances in split:', total)
        print('\n\n')

    print('Total number of instances in total:', all_total)

    # Plane and point counts
    for i in range(5):
        plane_name = plane_names[i] + '' * (20 - len(plane_names[i]))

        # Calculate correct numbers for inst
        all_inst_planes = 0
        all_inst_points = 0
        for i_inst in sem_label_to_inst[i]:
            all_inst_planes += sum_inst_planes[i_inst]
            all_inst_points += sum_inst_points[i_inst]

        print('%s & %i & %i & %i & %i \\\\' % (
            plane_name, sum_sem_planes[i], sum_sem_points[i], all_inst_planes, all_inst_points))
        print('\\hline')

    print('\nViz roofs:')
    for i in range(7):
        print('\t', sem_to_label[i], viz_roofs[i])

    '''
    npoints = 1024
    data = TR3DRoofsDataset(root=data_root, npoints=npoints, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for point_set, roof_type, seg in DataLoader:
        print('point set:', point_set.shape)
        print('roof type:', roof_type.shape)
        print('seg type :', seg.shape)
        break

    # Calcuate class ditribution
    seg_classes = {'Flat': [0], 'Hipped': [1], 'Gabled': [2], 'Corner Element': [3], 'T-Element': [4], 'Cross Element': [5], 'Combination': [6]}
    seg_label_to_cat = {}  # {0:Flat, 1:Hipped, ...6:Combination}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    class_distribution = {'Flat': 0, 'Hipped': 0, 'Gabled': 0, 'Corner Element': 0, 'T-Element': 0, 'Cross Element': 0, 'Combination': 0}

    for point_set, roof_type, seg in DataLoader:
        class_distribution[seg_label_to_cat[int(roof_type[0][0])]] += 1

    for cat in sorted(class_distribution.keys()):
        print('number of instances in class %s %f' % (cat + ' ' * (14 - len(cat)), class_distribution[cat]))
    
    '''
