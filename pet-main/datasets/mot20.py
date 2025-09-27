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
import configparser
from collections import defaultdict
from torch.utils.data import random_split

warnings.filterwarnings('ignore')

class mot20(Dataset):

    def __init__(self, data_root, transform=None, train=False, flip=False):
        
        self.root_path = data_root
        self.transform = transform
        self.train = train
        self.flip = flip
        self.patch_size = 256
        self.img_list = []   
        self.gt_list = {}    #  bbox + ids
        prefix = "train" if train else "val"
        self.prefix = prefix
        sequences = os.listdir(f"{data_root}/{prefix}")
        cls_people = {1,7}
        for seq in sequences:
            seq_path = os.path.join(data_root, prefix,seq)
            img_dir = os.path.join(seq_path, "img1")
            gt_path = os.path.join(seq_path, "gt", "gt.txt")
            
            seqinfo_path = os.path.join(seq_path, "seqinfo.ini")
            seqinfo =  parse_seqinfo(seqinfo_path)
           
            frame_gt_dict = defaultdict(list)
            if os.path.exists(gt_path):
                with open(gt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split(',')
                        frame, obj_id = int(parts[0]), int(parts[1])
                        x, y, w, h = map(float, parts[2:6])
                        conf = int(parts[6])
                        cls = int(parts[7])
                        if cls in cls_people:
                           bbox = [x, y, x + w, y + h]  #  x1, y1, x2, y2
                           frame_gt_dict[frame].append((bbox, obj_id))

            img_names = sorted(os.listdir(img_dir))
            for img_name in img_names:
                frame_id = int(img_name.split(".")[0])
                img_path = os.path.join(img_dir, img_name)
                self.img_list.append(img_path)
                self.gt_list[img_path] = frame_gt_dict.get(frame_id, [])  

        self.nSamples = len(self.img_list)
        print('Total MOT frames loaded:', self.nSamples)
    
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
        assert index < len(self), 'index range error'

        img_path = self.img_list[index]
        gt_info = self.gt_list[img_path]  # [(bbox, id), ...]
        
        img, points = load_data_mot20((img_path, gt_info), train=self.train)
        points = points.astype(float)  # N x 2, in [y, x] format
       
        # image transform（PIL → Tensor）
        if self.transform is not None:
            img = self.transform(img)
        img = torch.Tensor(img)

        if self.train:
            scale_range = [0.8, 1.2]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            if scale * min_size > self.patch_size:
                img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False).squeeze(0)
                points *= scale

        if self.train:
            img, points = random_crop(img, points, patch_size=self.patch_size)

        if random.random() > 0.5 and self.train and self.flip:
            img = torch.flip(img, dims=[2])  # Horizontal flip
            points[:, 1] = self.patch_size - points[:, 1]  

        target = {
            "points": torch.tensor(points, dtype=torch.float32),
            "labels": torch.ones((points.shape[0],), dtype=torch.long),
        }

        if self.train:
            density = self.compute_density(points)
            target["density"] = density

        if not self.train:
            target["image_path"] = img_path

        return img, target

def load_data_mot20(img_gt_tuple, train=False):
   
    img_path, gt_infos = img_gt_tuple
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    points = []

    for bbox, track_id in gt_infos:
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        head_x = x1 + w / 2
        head_y = y1 + h / 7  
        points.append([head_y, head_x])  
    points = np.array(points, dtype=np.float32)
    return img, points

def parse_seqinfo(seqinfo_path):
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    print("Reading:", seqinfo_path)
    print("Found sections:", config.sections())

    seq = config['Sequence']
    return {
        "name": seq.get('name'),
        "im_dir": seq.get('imDir'),
        "frame_rate": seq.getint('frameRate'),
        "seq_length": seq.getint('seqLength'),
        "width": seq.getint('imWidth'),
        "height": seq.getint('imHeight'),
        "im_ext": seq.get('imExt')
    }

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
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])

    data_root = args.data_path
    if image_set == 'train':
        train_set = mot20(data_root, train=True, transform=transform, flip=True)
        return train_set
    elif image_set == 'val':
        val_set = mot20(data_root, train=False, transform=transform)
        return val_set


def visualize_points_on_images(dataset, num_samples=5, save_dir="./vis_samples"):
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(dataset.img_list)), num_samples)

    for idx in indices:
        img_path = dataset.img_list[idx]
        gt_infos = dataset.gt_list[img_path]
        img, points = load_data_mot20((img_path, gt_infos), train=False)

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
