import torch
from MPM_Net import MPMNet
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import h5py


class trackp():
    def __init__(self, mag_th, itp, sigma, maxv, image_size):
        self.MAG_TH = mag_th
        self.ATP_UTV = itp
        self.SIGMA = sigma
        self.MAXV = maxv
        self.IMAGE_SIZE = image_size

    def loadModel(self, model_path, parallel=False):
        '''load UNet-like model
        Args:
             model_path (str): model path (.pth)
             parallel (bool): for multi-gpu
        Return:
            model
        '''
        
        model = MPMNet(6, 3)  

        model.cuda() 

        checkpoint = torch.load(model_path)
        if parallel:
            model = torch.nn.DataParallel(model)
        # model.load_state_dict(state_dict)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    # def load_fame0(self, coor_list_path):
    #     f = h5py.File(coor_list_path, 'r')
    #     coor_list = f['coordinate']
    #     fidt = f['d6']
    #     return coor_list, fidt

    def inferenceMPM(self, names, model):
        '''
        Args:
             names (list of str): list of image path
             model (nn.Module): torch model
        Return:
            mpm (numpy.ndarray): mpm (HEIGHT, WIDTH, 3), ch0: y_flow, ch1: x_flow, ch2: z_flow
        '''

        img1 = cv2.imread(names[0], -1)
        img2 = cv2.imread(names[1], -1)
        if len(img1.shape) == 2:
            img1 = np.expand_dims(img1, axis=2)
            img2 = np.expand_dims(img2, axis=2)
        img = np.concatenate([img1, img2], axis=-1)
        if img.max() > 1:
            img = img / 255
        img = img.transpose((2, 0, 1))

        img = torch.from_numpy(img).unsqueeze(0).type(torch.FloatTensor)
        print(img.shape)
        img = img.cuda()

        output = model(img)
        mpm = output[0].cpu().detach().numpy()
        mpm = np.transpose(mpm, axes=[1, 2, 0])
        return mpm

    def getMPMpoints(self, mpm, mag, mag_max=1.0):
        '''
        Args:
            mpm (numpy.ndarray): MPM
            mag (numpy.ndarray): Magnitude of MPM i.e. heatmap
            mag_max: Maximum value of heatmap
        Return:
            result (numpy.ndarray): Table of peak coordinates, warped coordinates, and peak value [x, y, wx, wy, pv]
        '''
        mag[mag > mag_max] = mag_max
        map_left_top, map_top, map_right_top = np.zeros(mag.shape), np.zeros(mag.shape), np.zeros(mag.shape)
        map_left, map_right = np.zeros(mag.shape), np.zeros(mag.shape)
        map_left_bottom, map_bottom, map_right_bottom = np.zeros(mag.shape), np.zeros(mag.shape), np.zeros(mag.shape)
        map_left_top[1:, 1:], map_top[:, 1:], map_right_top[:-1, 1:] = mag[:-1, :-1], mag[:, :-1], mag[1:, :-1]
        map_left[1:, :], map_right[:-1, :] = mag[:-1, :], mag[1:, :]
        map_left_bottom[1:, :-1], map_bottom[:, :-1], map_left_bottom[1:, :-1] = mag[:-1, 1:], mag[:, 1:], mag[1:, 1:]
        peaks_binary = np.logical_and.reduce((
            mag >= map_left_top, mag >= map_top, mag >= map_right_top,
            mag >= map_left, mag > self.MAG_TH, mag >= map_right,
            mag >= map_left_bottom, mag >= map_bottom, mag >= map_right_bottom,
        ))
        _, _, _, center = cv2.connectedComponentsWithStats((peaks_binary * 1).astype('uint8'))
        center = center[1:]

        result = []
        for center_cell in center.astype('int'):
            result.append([center_cell[0], center_cell[1]])
        return result



    def motion_associate(self,dets_one, trks,track_indices,mag,distance_threshold):
        
        pred_pos = [traj.predicted_coordinate for traj in trks.values()]
        pos = dets_one["coordinate"]

        # DISTANCE_THRESHOLD = 10.0
        distance_matrix = self.get_motion_scores(pred_pos, pos, mag, distance_threshold = 10)
        modified_matrix = np.sqrt(distance_matrix)
        
        row_indices, col_indices = linear_sum_assignment(-distance_matrix)  
        
        matched_pairs = []
        for det_idx, trk_idx in zip(row_indices, col_indices):
            score = modified_matrix[det_idx, trk_idx]
            if score > 0.5:  
                matched_pairs.append((det_idx, track_indices[trk_idx], score))
        print(f'# motions_match of associated people: {len(matched_pairs)}')
        
        matched_pairs = np.array(matched_pairs, dtype=float) 
        unmatched_detections = []

        if matched_pairs.shape[0] == 0:  #no matched
            unmatched_detections = list(range(len(pos)))
            unmatched_trackers  = list(trks.keys())
            matched_pairs = np.empty((0, 3), dtype=float)
        else:

            matched_det_indices = set(matched_pairs[:, 0].astype(int))
            unmatched_detections = [d for d in range(len(pos)) if d not in matched_det_indices]

            matched_trk_indices = set(matched_pairs[:, 1].astype(int))
            unmatched_trackers = [t for t in trks.keys() if t not in matched_trk_indices]

        return (matched_pairs,np.array(unmatched_detections),np.array(unmatched_trackers)),modified_matrix

    def get_motion_scores(self,pred_pos, pos, mag, distance_threshold):

        DISTANCE_THRESHOLD = distance_threshold  
        distance_matrix = np.zeros((len(pos), len(pred_pos)))
        print(f"motions_match scores------len(pos): {len(pos)},len(pred_pos):{len(pred_pos)}")
        num = 0
        for i, focus_pos in enumerate(pred_pos):
            x,y = focus_pos[0],focus_pos[1]
            distances = np.linalg.norm(pos[:, 0:2] - np.array([[x, y]]), axis=1)
            num += 1
            scores = np.where(distances < DISTANCE_THRESHOLD,
                                (DISTANCE_THRESHOLD - distances) / DISTANCE_THRESHOLD,
                                0)
                                
            distance_matrix[:, i] = scores
        return distance_matrix
