import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import torchvision.transforms as standard_transforms
import warnings

warnings.filterwarnings('ignore')


class DroneBird(Dataset):

    def __init__(self, data_root, transform=None, train=False, flip=False):
        self.root_path = data_root
        # prefix = "train_data" if train else "val_data"
        prefix = "train" if train else "val"
        self.prefix = prefix
        self.img_list = os.listdir(f"{data_root}/{prefix}/images")

        # get image and ground-truth list
        self.gt_list = {}
        for img_name in self.img_list:
            img_path = f"{data_root}/{prefix}/images/{img_name}"
            gt_path = f"{data_root}/{prefix}/ground_truth/GT_{img_name}"
            self.gt_list[img_path] = gt_path.replace("jpg", "mat")
        self.img_list = sorted(list(self.gt_list.keys()))

        subset_ratio = 0.5
        # ===== random extract subset_ratio sampling =====
        if train and 0 < subset_ratio < 1.0:
            n_samples = int(len(self.img_list) * subset_ratio)
            self.img_list = random.sample(self.img_list, n_samples)
            self.gt_list = {img: self.gt_list[img] for img in self.img_list}

        self.nSamples = len(self.img_list)
        print('nSamples:' + str(self.nSamples))

        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = 256

    def compute_density(self, points):
        """
        Compute crowd density:
            - defined as the average nearest distance between ground-truth points
        """
        points_tensor = torch.from_numpy(points.copy())
        dist = torch.cdist(points_tensor, points_tensor, p=2)
        if points_tensor.shape[0] > 1:
            density = dist.sort(dim=1)[0][:, 1].mean().reshape(-1)
        else:
            density = torch.tensor(999.0).reshape(-1)
        return density

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        # load image and gt points
        img_path = self.img_list[index]
        gt_path = self.gt_list[img_path]
        img, points = load_data((img_path, gt_path), self.train)
        points = points.astype(float)

        # image transform
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        # random scale
        if self.train:
            scale_range = [0.8, 1.2]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)

            # interpolation
            if scale * min_size > self.patch_size:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                points *= scale

        # random crop patch
        if self.train:
            img, points = random_crop(img, points, patch_size=self.patch_size)

        # random flip
        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])
            points[:, 1] = self.patch_size - points[:, 1]

        # target
        # print(points)
        target = {}
        target['points'] = torch.Tensor(points)
        target['labels'] = torch.ones([points.shape[0]]).long()

        if self.train:
            density = self.compute_density(points)
            target['density'] = density

        if not self.train:
            target['image_path'] = img_path

        return img, target


def load_data(img_gt_path, train):

    img_path, gt_path = img_gt_path
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    mat = io.loadmat(gt_path)
    points = mat['locations'][:, :2][:, ::-1] #this is reverse, maybe dataset sets

    return img, points


def random_crop(img, points, patch_size=256):
    patch_h = patch_size
    patch_w = patch_size

    # random crop
    start_h = random.randint(0, img.size(1) - patch_h) if img.size(1) > patch_h else 0
    start_w = random.randint(0, img.size(2) - patch_w) if img.size(2) > patch_w else 0
    end_h = start_h + patch_h
    end_w = start_w + patch_w
    idx = (points[:, 0] >= start_h) & (points[:, 0] <= end_h) & (points[:, 1] >= start_w) & (points[:, 1] <= end_w)

    # clip image and points
    result_img = img[:, start_h:end_h, start_w:end_w]
    result_points = points[idx]
    result_points[:, 0] -= start_h
    result_points[:, 1] -= start_w

    # resize to patchsize
    imgH, imgW = result_img.shape[-2:]
    fH, fW = patch_h / imgH, patch_w / imgW
    result_img = torch.nn.functional.interpolate(result_img.unsqueeze(0), (patch_h, patch_w)).squeeze(0)
    result_points[:, 0] *= fH
    result_points[:, 1] *= fW
    return result_img, result_points


def build(image_set, args):
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],                                                                std=[0.229, 0.224, 0.225]),
    ])
    data_root = args.data_path
    if image_set == 'train':
        train_set = DroneBird(data_root, train=True, transform=transform, flip=True)
        return train_set
    elif image_set == 'val':
        val_set = DroneBird(data_root, train=False, transform=transform)
        return val_set

def visualize_points_on_images(dataset, num_samples=5, save_dir="./vis_samples"):
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset.img_list)), num_samples)

    for idx in indices:
        img_path = dataset.img_list[idx]
        gt_infos = dataset.gt_list[img_path]
        img, points = load_data((img_path, gt_infos), train=False)

        img_draw = np.array(img).copy()
        for pt in points:
            y, x = int(pt[0]), int(pt[1])
            cv2.circle(img_draw, (x, y), 4, (0, 255, 255), -1)
            dy = 20
            dx = 20
            x1 = x - dx
            y1 = y - dy
            x2 = x + dx
            y2 = y + dy
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        save_name = os.path.basename(img_path).replace('.jpg', '_vis.jpg')
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))  


# 1. load
dataset = DroneBird(data_root="./tracking/DroneBird", train=True, flip=False)

# #
# val_ratio = 0.2
# n_val = int(len(dataset_train_total) * val_ratio)
# n_train = len(dataset_train_total) - n_val
# dataset_train, dataset_val = random_split(dataset_train_total, [n_train, n_val])
# dataset_val = build_dataset(image_set='val', args=args)

# 2. visual
output_dir = "./pet-main/dataset_visual"
save_dir = os.path.join(output_dir,"crop_dronebird")
# print("the size train ,the size val",len(dataset_train),len(dataset_val))

os.makedirs(save_dir, exist_ok=True)
visualize_points_on_images(dataset, num_samples=5, save_dir = save_dir)
