# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

class TR3DRoofsDataset(Dataset):
    def __init__(self, root='./data/TR3DRoofs', npoints=2500, split='train', seg_type="inst"): # inst_type: "sem" | "isnt"
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

        data_dir = os.path.join(self.root, '00000000')
        self.fns = sorted(os.listdir(data_dir)) # file names

        # fn[0:-4] is file extention '.txt' so remove it
        if split == 'trainval':
            self.fns = [fn for fn in self.fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
        elif split == 'train':
            self.fns = [fn for fn in self.fns if fn[0:-4] in train_ids]
        elif split == 'val':
            self.fns = [fn for fn in self.fns if fn[0:-4] in val_ids]
        elif split == 'test':
            self.fns = [fn for fn in self.fns if fn[0:-4] in test_ids]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)

        # Add dir path to file names
        self.fns = [os.path.join(data_dir, fn) for fn in self.fns]

        # Define the plane categories in self.roof_types
        self.roof_types = {'Flat': 1, 'Hipped': 2, 'Gabled': 3, 'Corner Element': 4, 'T-Element': 5, 'Cross Element': 6, 'Combination': 7}

        self.plane_categories = {'rectangular': [1, 2], 'trapezoid': [
            2, 3], 'triangular': [4, 5], 'parallelogram': [6, 7], 'ladder': [8, 9, 10, 11]}

        # Store some points in a cache
        self.cache = {}  # from index to (point_set, roof_type, seg) tuple
        self.cache_size = 20000

    
    def __getitem__(self, index):
        if index in self.cache:
            point_set, roof_type, seg = self.cache[index]
        else:
            # Data structure
            # x, y, z, roof type, semantic label, instance label
            # xyz is normalized for the whole roof
            data = np.loadtxt(self.fns[index]).astype(np.float32)
            point_set = data[:, 0:3] # already normalized
            roof_type = data[:, 3].astype(np.int32)
            sem_seg = data[:, 4].astype(np.int32)
            inst_seg = data[:, 5].astype(np.int32)

            # TODO: only store the last cache_size points
            if len(self.cache) < self.cache_size:
                if self.seg_type == "inst":
                    self.cache[index] = (point_set, roof_type, inst_seg)
                else:
                    self.cache[index] = (point_set, roof_type, sem_seg)
        
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
        roof_type = roof_type[choice]
        sem_seg = sem_seg[choice]
        inst_seg = inst_seg[choice]

        if self.seg_type == "inst":
            return point_set, roof_type, inst_seg
        return point_set, roof_type, sem_seg
        

    def __len__(self):
        return len(self.fns)

# Test implementation
if __name__ == '__main__':
    import torch

    data_root = 'data/tr3d_roof_segmented_dataset/'
    npoints = 1024
    data = TR3DRoofsDataset(root=data_root, npoints=npoints, split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point_set, roof_type, inst_seg in DataLoader:
        print(point_set.shape)
        print(roof_type.shape)
        print(inst_seg.shape)
