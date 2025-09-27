import os.path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

MAG_TH = 0.3

def Get_MPM_Number(mpms):
    mag_max = 1.0
    count = 0
    mpms = mpms.cpu()
    for m_id in range(len(mpms)):
        mag = mpms[m_id].pow(2).sum(dim=0, keepdim=True).sqrt()

        mag = mag.cpu()
        mag = mag[0].numpy()
        # mag = gaussian_filter(mag, sigma=3, mode='constant')
        # print(mag.shape)
        mag[mag > mag_max] = mag_max
        map_left_top, map_top, map_right_top = np.zeros(mag.shape), np.zeros(mag.shape), np.zeros(mag.shape)
        map_left, map_right = np.zeros(mag.shape), np.zeros(mag.shape)
        map_left_bottom, map_bottom, map_right_bottom = np.zeros(mag.shape), np.zeros(mag.shape), np.zeros(mag.shape)
        map_left_top[1:, 1:], map_top[:, 1:], map_right_top[:-1, 1:] = mag[:-1, :-1], mag[:, :-1], mag[1:, :-1]
        map_left[1:, :], map_right[:-1, :] = mag[:-1, :], mag[1:, :]
        map_left_bottom[1:, :-1], map_bottom[:, :-1], map_left_bottom[1:, :-1] = mag[:-1, 1:], mag[:, 1:], mag[1:, 1:]
        peaks_binary = np.logical_and.reduce((
            mag >= map_left_top, mag >= map_top, mag >= map_right_top,
            mag >= map_left, mag > MAG_TH, mag >= map_right,
            mag >= map_left_bottom, mag >= map_bottom, mag >= map_right_bottom,
        ))
        _, _, _, center = cv2.connectedComponentsWithStats((peaks_binary * 1).astype('uint8'))
        center = center[1:]
        count += len(center)
    return count



def Save_Show(mpms_gt, mpms_pred, path):
   
    mpms_gt = mpms_gt.cpu()
    mpms_pred = mpms_pred.cpu()
    for m_id in range(len(mpms_gt)):
        mag_gt = mpms_gt[m_id].pow(2).sum(dim=0, keepdim=True).sqrt()
        mag_gt = mag_gt.cpu()
        mag_gt = mag_gt[0].numpy()

        mag_pred = mpms_pred[m_id].pow(2).sum(dim=0, keepdim=True).sqrt()
        mag_pred = mag_pred.cpu()
        mag_pred = mag_pred[0].numpy()

        mag_gt = mag_gt / np.max(mag_gt) * 255
        map_gt = mag_gt.astype(np.uint8)
        map_gt = cv2.applyColorMap(map_gt, 2)

        mag_pred = mag_pred / np.max(mag_pred) * 255
        map_pred = mag_pred.astype(np.uint8)
        map_pred = cv2.applyColorMap(map_pred, 2)

        result_img = np.hstack((mag_gt, mag_pred))

        cv2.imwrite(os.path.join(path, 'mag'+str(m_id)+'.jpg'), result_img)



def eval_net(net, loader, device, criterion, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    total_error = 0
    mae = 0.0

    root_dir = '/home/zk/charac_pos/MPM-master/mot17-eval'
    sub = 'epoch'+str(global_step)
    sub_root_dir = os.path.join(root_dir, sub)
    if not os.path.exists(sub_root_dir):
        os.makedirs(sub_root_dir)

    
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for i, batch in enumerate(loader):
            imgs, mpms_gt = batch['img'], batch['mpm']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mpms_gt = mpms_gt.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mpms_pred = net(imgs)

            total_error += criterion(mpms_pred, mpms_gt).item()

            gt_count = Get_MPM_Number(mpms_gt)
            count = Get_MPM_Number(mpms_pred)
            
            subsub = 'batch'+str(i)
            subsub_root_dir = os.path.join(sub_root_dir, subsub)
            if not os.path.exists(subsub_root_dir):
                os.makedirs(subsub_root_dir)

            Save_Show(mpms_gt, mpms_pred, subsub_root_dir)
            
            mae += abs(gt_count - count)/len(batch['img'])

            pbar.update()

    net.train()
    return total_error / n_val, mae*1.0/n_val