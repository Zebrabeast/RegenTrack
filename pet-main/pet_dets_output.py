import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as standard_transforms

import util.misc as utils
from models import build_model
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    
    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)       # classification loss coefficient
    parser.add_argument('--point_loss_coef', default=5.0, type=float)    # regression loss coefficient
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")   # cross-entropy weights

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)

    # misc parameters
    parser.add_argument('--device', default='cuda',    
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--vis_dir', default="")
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def visualization(samples, pred, vis_dir, img_path, split_map=None):
    """
    Visualize predictions
    """

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # record [p_id,x,y] for save txt file
        person_id = 0
        size = 3
        pred_positions = []  # Store prediction positions for saving to txt file
        # draw predictions (green)
        for p in pred[idx]:
            if isinstance(p, torch.Tensor):
                p = p.tolist()  #  Python list

            if isinstance(p, (list, tuple)) and len(p) >= 2:                
                y, x = int(p[0]), int(p[1])
                sample_vis = cv2.circle(sample_vis, (x, y), size, (0, 255, 0), -1)
                pred_positions.append([person_id, x, y])
                person_id += 1

        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis

        # save image
        if vis_dir is not None:
            # Create two folders to facilitate partition management of the two outputs

            # save visualization image
            # vis_img_root = os.path.join(vis_dir, 'vis_img')
            # if not os.path.exists(vis_img_root):
            #     os.makedirs(vis_img_root)

            # save individual coordinate
            txt_root = os.path.join(vis_dir, 'txt')
            if not os.path.exists(txt_root):
                os.makedirs(txt_root)

            # # eliminate invalid area  
            # imgH, imgW = masks.shape[-2:]
            # valid_area = torch.where(~masks[idx])
            # valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            # sample_vis = sample_vis[:valid_h + 1, :valid_w + 1]

            name = img_path.split('/')[-1].split('.')[0]
            # cv2.imwrite(os.path.join(vis_img_root, '{}_pred_{}.jpg'.format(name, len(pred[idx]))),
            #             sample_vis)

            # save predictions to txt file
            txt_path = os.path.join(txt_root, '{}_pred_positions.txt'.format(name))
            with open(txt_path, 'w') as f:
                for person in pred_positions:
                    f.write('{},{},{}'.format(person[0], person[1], person[2]))
                    f.write('\n')

def nms(points, scores, dist_thresh):
   
    device = points.device if isinstance(points, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32, device=device)
    else:
        scores = scores.clone().detach().to(dtype=torch.float32, device=device)

    
    assert points.ndim == 2 and points.size(1) == 2, "points must be a tensor of shape [N, 2]"
    assert scores.ndim == 1, "scores must be a 1D tensor with one score per point"

   
    if points.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    
    sorted_indices = torch.argsort(scores, descending=True).to(device)

    keep_indices = []
    while sorted_indices.numel() > 0:
      
        current_index = sorted_indices[0]
        keep_indices.append(current_index)

        current_point = points[current_index]

        remaining_points = points[sorted_indices]
        if remaining_points.shape[0] > 1:
            distances = torch.cdist(current_point.view(1, -1), remaining_points).squeeze(0)
        else:
            distances = torch.tensor([0.0], device=device)
        mask = distances > dist_thresh
        sorted_indices = sorted_indices[mask]

    return points[torch.tensor(keep_indices, device=device, dtype=torch.long)]

def dynamic_nms(points, scores, dist_base=12, alpha=0.3, density_map=None):
    device = points.device if isinstance(points, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32, device=device)
    else:
        scores = scores.clone().detach().to(dtype=torch.float32, device=device)

    assert points.ndim == 2 and points.size(1) == 2, "points must be of shape [N, 2]"
    assert scores.ndim == 1, "scores must be 1D tensor"

    if points.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    sorted_indices = torch.argsort(scores, descending=True).to(device)
    keep_indices = []

    while sorted_indices.numel() > 0:
        current_index = sorted_indices[0]
        keep_indices.append(current_index)
        current_point = points[current_index]

        remaining_indices = sorted_indices[1:]
        if remaining_indices.numel() == 0:
            break
        remaining_points = points[remaining_indices]

        distances = torch.cdist(current_point.view(1, -1), remaining_points).squeeze(0)  # [M]


        if density_map is not None:

            dynamic_thresh = []
            for pt in remaining_points:
                x, y = int(pt[0].item()), int(pt[1].item())
                local = density_map[max(0, y-3):y+4, max(0, x-3):x+4]
                local_density = local.sum().item()
                r = dist_base / (1 + alpha * local_density)
                dynamic_thresh.append(r)
            dynamic_threshs = torch.tensor(dynamic_thresh, device=device)
        else:
           
            current_score = scores[current_index]
            remaining_scores = scores[remaining_indices]
            dynamic_threshs = dist_base / (1 + alpha * remaining_scores)

       
        mask = distances > dynamic_threshs
        sorted_indices = sorted_indices[1:][mask]  

    return points[torch.tensor(keep_indices, device=device, dtype=torch.long)]

def dynamic_nms_scores_before(points, scores, dist_base=12, alpha=0.3, density_map=None):
    device = points.device if isinstance(points, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32, device=device)
    else:
        scores = scores.clone().detach().to(dtype=torch.float32, device=device)

    assert points.ndim == 2 and points.size(1) == 2, "points must be of shape [N, 2]"
    assert scores.ndim == 1, "scores must be 1D tensor"

    if points.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    # Step 1: 
    high_score_mask = scores > 0.8
    high_score_points = points[high_score_mask]

    # Step 2: 
    remaining_points = points[~high_score_mask]
    remaining_scores = scores[~high_score_mask]

    if remaining_points.shape[0] == 0:
        return high_score_points 

    sorted_indices = torch.argsort(remaining_scores, descending=True).to(device)
    keep_indices = []

    while sorted_indices.numel() > 0:
        current_index = sorted_indices[0]
        keep_indices.append(current_index)
        current_point = remaining_points[current_index]

        remaining_indices = sorted_indices[1:]
        if remaining_indices.numel() == 0:
            break
        rem_points = remaining_points[remaining_indices]

        distances = torch.cdist(current_point.view(1, -1), rem_points).squeeze(0)

        if density_map is not None:
            dynamic_thresh = []
            for pt in rem_points:
                x, y = int(pt[0].item()), int(pt[1].item())
                local = density_map[max(0, y-3):y+4, max(0, x-3):x+4]
                local_density = local.sum().item()
                r = dist_base / (1 + alpha * local_density)
                dynamic_thresh.append(r)
            dynamic_threshs = torch.tensor(dynamic_thresh, device=device)
        else:
            current_score = remaining_scores[current_index]
            rem_scores = remaining_scores[remaining_indices]
            dynamic_threshs = dist_base / (1 + alpha * rem_scores)

        mask = distances > dynamic_threshs
        sorted_indices = remaining_indices[mask]

    selected_points = remaining_points[torch.tensor(keep_indices, device=device, dtype=torch.long)]

    final_points = torch.cat([high_score_points, selected_points], dim=0)

    return final_points

def dynamic_nms_scores_after(points, scores, dist_base=12, alpha=0.3, density_map=None):
    device = points.device if isinstance(points, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    points = torch.tensor(points, dtype=torch.float32, device=device)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32, device=device)
    else:
        scores = scores.clone().detach().to(dtype=torch.float32, device=device)

    assert points.ndim == 2 and points.size(1) == 2, "points must be of shape [N, 2]"
    assert scores.ndim == 1, "scores must be 1D tensor"

    if points.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)

    sorted_indices = torch.argsort(scores, descending=True).to(device)
    keep_indices = []

    while sorted_indices.numel() > 0:
        current_index = sorted_indices[0]
        keep_indices.append(current_index)
        current_point = points[current_index]

        remaining_indices = sorted_indices[1:]

        if remaining_indices.numel() == 0:
            break
        remaining_points = points[remaining_indices]

       
        distances = torch.cdist(current_point.view(1, -1), remaining_points).squeeze(0)  # [M]
    
        if density_map is not None:
     
            dynamic_thresh = []
            for pt in remaining_points:
                x, y = int(pt[0].item()), int(pt[1].item())
                local = density_map[max(0, y-3):y+4, max(0, x-3):x+4]
                local_density = local.sum().item()
                # print('local_density为',local_density)
                r = dist_base / (1 + alpha * local_density)
                dynamic_thresh.append(r)
            dynamic_threshs = torch.tensor(dynamic_thresh, device=device)
        else:
           
            current_score = scores[current_index]
            remaining_scores = scores[remaining_indices]
            dynamic_threshs = dist_base / (1 + alpha * remaining_scores)

       
        close_mask = distances <= dynamic_threshs
        rem_scores =  scores[remaining_indices]
        low_score_mask = rem_scores <= 0.8
        suppress_mask = close_mask & low_score_mask  
        mask = ~suppress_mask  
        sorted_indices = remaining_indices[mask]

    return points[torch.tensor(keep_indices, device=device, dtype=torch.long)]

@torch.no_grad()
def evaluate_single_image(model, img_path, device, vis_dir=None):
    model.eval()

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    # load image
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # transform image
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = torch.Tensor(img)
    samples = utils.nested_tensor_from_tensor_list([img])
    samples = samples.to(device)
    img_h, img_w = samples.tensors.shape[-2:]

    # inference
    outputs = model(samples, test=True)
    raw_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)
    outputs_scores = raw_scores[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]
    print('prediction: ', len(outputs_scores))
    
    # visualize predictions
    if vis_dir: 
        points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
        density_map = generate_density_map_fast(points,img_shape)
        # dist_thresh = 4.0  
        # if len(outputs_scores) == 0:
        #     kept_points = points
        # else:
            # kept_points = nms(points, outputs_scores, dist_thresh)
            # kept_points = dynamic_nms(points, outputs_scores, dist_base = 8,density_map=density_map)
        kept_points = points
       
        split_map = (outputs['split_map_raw'][0].detach().squeeze(0) > 0.5).float().cpu().numpy()
        visualization(samples, [kept_points], vis_dir, img_path, split_map=split_map)

def generate_density_map_fast(pred_points, img_shape, sigma=3, kernel_size=15):
    H, W = img_shape
    density_map = np.zeros((H, W), dtype=np.float32)

    half_k = kernel_size // 2
    gauss_kernel = gaussian_kernel_patch(kernel_size, sigma)

    for x, y in pred_points:
        x, y = int(x), int(y)

      
        x1, y1 = max(0, x - half_k), max(0, y - half_k)
        x2, y2 = min(W, x + half_k + 1), min(H, y + half_k + 1)

      
        kx1, ky1 = half_k - (x - x1), half_k - (y - y1)
        kx2, ky2 = kx1 + (x2 - x1), ky1 + (y2 - y1)
        if y2 > y1 and x2 > x1 and ky2 > ky1 and kx2 > kx1:
            density_map[y1:y2, x1:x2] += gauss_kernel[ky1:ky2, kx1:kx2]

    return density_map

def gaussian_kernel_patch(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel.astype(np.float32)

def process_sequence(sequence_path, output_dir, model, device): 
    image_files = sorted([
        f for f in os.listdir(sequence_path) if f.endswith(('.jpg', '.png', '.jpeg'))
    ])

    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(sequence_path)}"):
        img_path = os.path.join(sequence_path, img_file)
        evaluate_single_image(model, img_path, device, vis_dir=output_dir)

def main(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Building model...")
    model, criterion = build_model(args)
    model.to(device)
    model.eval()

    #  checkpoint
    print(f"Loading checkpoint from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model'])


    sequence_dirs = sorted([
        os.path.join(args.root_dir, d) for d in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, d))
    ])

    for seq_dir in sequence_dirs:
        seq_name = os.path.basename(seq_dir)
        output_dir = os.path.join(args.output_root, seq_name)
        os.makedirs(output_dir, exist_ok=True)
        origin_dir = os.path.join(seq_dir, "origin")
        # origin_dir = os.path.join(seq_dir, "img1")
        
        print(f"\n==> Start processing sequence: {seq_name}")
        try:
            process_sequence(origin_dir, output_dir, model, device)
        except Exception as e:
            print(f"[ERROR] Failed to process {seq_name}: {e}")
        torch.cuda.empty_cache() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET batch evaluation script', parents=[get_args_parser()])
    parser.add_argument('--root_dir', type=str, default='/home/zk/data_SSD/zk/test_data', 
                        help="Root folder containing multiple video sequences (each as a folder)")
    parser.add_argument('--output_root', type=str, default='/home/zk/data_SSD/zk/pet_outputs/test_data', 
                        help="Output folder to store visual results per sequence")
    parser.add_argument('--resume',default='DroneCrowd_weight/best_checkpoint.pth', 
                        help="the weight of  pet model")
    args = parser.parse_args()
    main(args)