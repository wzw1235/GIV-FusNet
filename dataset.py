import os

import numpy as np
import tifffile
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize


class_dic = {
    "transport": 0,
    "industrial area": 1,
    "office building": 2,
    "sport field": 3,
    "farmland": 4,
    "forest": 5,
    "residential area": 6,
    "residential": 6,
    "church": 7,
    "meadow": 8,
}


class SceneDataset(Dataset):
    def __init__(self, data_path: str, voxel_size: float):
        self.data_path = data_path
        self.voxel_size = voxel_size
        self.img_list = []
        self.voxel_list = []
        self.feat_list = []
        self.label = []

        for class_name in sorted(os.listdir(data_path)):
            if class_name not in class_dic:
                continue

            class_path = os.path.join(data_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for folder in sorted(os.listdir(class_path)):
                folder_path = os.path.join(class_path, folder)
                if not os.path.isdir(folder_path):
                    continue

                point_path = os.path.join(folder_path, "Point.npy")
                img_path = os.path.join(folder_path, "img.tiff")

                if not os.path.exists(point_path) or not os.path.exists(img_path):
                    continue

                points = np.load(point_path)
                points -= np.mean(points, axis=0)
                points /= np.max(np.sqrt(np.sum(points ** 2, axis=1)))

                coords = points[:, :3]
                coords -= np.min(coords, axis=0, keepdims=True)
                coords, indices = sparse_quantize(
                    coords, voxel_size, return_index=True
                )

                self.voxel_list.append(coords)
                self.feat_list.append(points[indices])
                self.label.append(class_dic[class_name])

                img = tifffile.imread(img_path).astype(np.uint8)
                img = Image.fromarray(img.transpose(1, 2, 0))
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                img = np.asarray(img).transpose(2, 0, 1)
                self.img_list.append(img)

    def __getitem__(self, index):
        point = SparseTensor(
            coords=torch.tensor(self.voxel_list[index], dtype=torch.int),
            feats=torch.tensor(self.feat_list[index], dtype=torch.float),
        )

        return {
            "point": point,
            "image": torch.tensor(self.img_list[index], dtype=torch.float),
            "label": torch.tensor(self.label[index], dtype=torch.long),
        }

    def __len__(self):
        return len(self.img_list)