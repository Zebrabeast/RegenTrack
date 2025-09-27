from mpm_track_parts_plus import trackp
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn import preprocessing
from scipy.sparse import csr_matrix
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import os
import colorsys
import random
from glob import glob
import torch

import time
import argparse
import pickle
from tqdm import tqdm
from knn import KNN
from feature_extractor import ReIDFeatureExtractor
from diffusion import Diffusion

Candidate_pool = {}    
can_traj = 0           
can_len = 0

# global parameter
actual_total_frames = 0 


class Trajectory:
    def __init__(self, obj_id, feature, coordinate, max_len= 8, decay=0.87):
        self.obj_id = obj_id
        self.features = [feature]  
        self.current_coordinate = coordinate  
        self.predicted_coordinate = None  
        self.max_len = max_len  
        self.decay = decay  
        self.unmatched_act = 0
 
    def update(self, new_feature, new_coordinate=None):
        if new_feature is not None:
            self.features.append(new_feature)
            if len(self.features) > self.max_len:
                self.features.pop(0)
            self.unmatched_act = 0 
        else:
            self.unmatched_act += 1 

        if new_coordinate is not None:
            self.current_coordinate = new_coordinate

    def predict(self, mpm):
        pos = self.current_coordinate
        
        pos_x = int(round(pos[0]))
        pos_y = int(round(pos[1]))
        
        # boundary check
        height, width = 1080,1920  
        pos_x = max(0, min(pos_x, width - 1))  
        pos_y = max(0, min(pos_y, height - 1))
        
        vec = mpm[pos_y, pos_x]
        # mag_value = pre_mag[center_cell[1], center_cell[0]]
        vec = vec / np.linalg.norm(vec)
        if abs(vec[2]) < 0.9:
            vec[2] = 0.98
        x = 10000 if vec[2] == 0 else int(5 * (vec[1] / vec[2]))  # Calculate the motion vector x
        y = 10000 if vec[2] == 0 else int(5 * (vec[0] / vec[2]))  # Calculate the motion vector y
        pred_coord = [pos[0] + x, pos[1] + y] 

        self.predicted_coordinate = pred_coord

    def get_weighted_feature(self):
        """计算加权平均特征"""
        weights = np.array([self.decay ** i for i in range(len(self.features))][::-1])  # Decreasing weights
        weights /= np.sum(weights)  
        return np.sum(np.array(self.features) * weights[:, None], axis=0)

    def get_position(self):
        """返回当前和预测坐标"""
        return {
            "current": self.current_coordinate,
            "predicted": self.predicted_coordinate
        }
 
class Candidate_traj:
    def __init__(self, obj_id, feature, frame_id, coordinate, max_len=10, decay=0.87):
        self.obj_id = obj_id
        self.history = []
        self.hits = 1    
        self.age = 1      
        self.max_len = max_len  
        self.decay = decay
        self.last_update_frame = frame_id
        self.current_coordinate = coordinate
        self.predicted_coordinate = None
        self.history.append({   
            'frame': frame_id,
            'coord': coordinate,
            'feature': feature
        })

    def add_record(self, frame_id, feature, coordinate):
        self.history.append({
            'frame': frame_id,
            'coord': coordinate,
            'feature': feature
        })
        self.hits += 1
        self.age += 1
        self.current_coordinate = coordinate
        self.last_update_frame = frame_id
        
    def get_weighted_feature(self):
        
        features = [x['feature'] for x in self.history[-8:]]
        weights = np.array([self.decay ** i for i in range(len(features))][::-1])  
        weights /= np.sum(weights)  
        return np.sum(np.array(features) * weights[:, None], axis=0)

    def predict(self, mpm): 
        pos = self.current_coordinate
       
        pos_x = int(round(pos[0]))
        pos_y = int(round(pos[1]))
        
        height, width = 1080,1920 
        pos_x = max(0, min(pos_x, width - 1))   
        pos_y = max(0, min(pos_y, height - 1)) 
        
        vec = mpm[pos_y, pos_x]
        # mag_value = pre_mag[center_cell[1], center_cell[0]]
        vec = vec / np.linalg.norm(vec)
        if abs(vec[2]) < 0.9:
            vec[2] = 0.98
        x = 10000 if vec[2] == 0 else int(5 * (vec[1] / vec[2]))  
        y = 10000 if vec[2] == 0 else int(5 * (vec[0] / vec[2]))  
        pred_coord = [pos[0] + x, pos[1] + y]  
        
        self.predicted_coordinate = pred_coord

def clean_candidate_pool(candidate_pool, current_frame,
                         max_inactive_frames= 20, 
                         min_confirm_hits=16,
                         age_threshold=20):
    to_remove = []   
    for obj_id, traj in candidate_pool.items():
        inactive_time = current_frame - traj.last_update_frame
        if inactive_time > max_inactive_frames:
            to_remove.append(obj_id)
        elif traj.hits < min_confirm_hits and traj.age > age_threshold:
            to_remove.append(obj_id)
        elif traj.hits >= min_confirm_hits:
            to_remove.append(obj_id)
    print("已经删除了的候选池的数量",len(to_remove))
    for obj_id in to_remove:
        del candidate_pool[obj_id]

def read_coords_from_pet(file_path):
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    return coords

def reset_globals():
    global Candidate_pool, can_traj,can_len 
    Candidate_pool = {}
    can_traj = 0          
    can_len = 0

def fuse_detections_weighted(det1, det2, threshold=5, weight1=0.7, weight2=0.3):

    det1 = np.array(det1)
    det2 = np.array(det2)

    if len(det1) == 0:
        return [[i, *coord] for i, coord in enumerate(det2)]
    if len(det2) == 0:
        return [[i, *coord] for i, coord in enumerate(det1)]

    dists = cdist(det1, det2)
    matched_1 = set()
    matched_2 = set()
    fused = []

    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            if dists[i, j] < threshold and i not in matched_1 and j not in matched_2:
                x = weight1 * det1[i][0] + weight2 * det2[j][0]
                y = weight1 * det1[i][1] + weight2 * det2[j][1]
                fused.append([x, y])
                matched_1.add(i)
                matched_2.add(j)

    for i in range(len(det1)):
        if i not in matched_1:
            fused.append(det1[i].tolist())
    for j in range(len(det2)):
        if j not in matched_2:
            fused.append(det2[j].tolist())

    fused_with_id = [[i, coord[0], coord[1]] for i, coord in enumerate(fused)]
    return fused_with_id

def mpm_track(seq_dir, save_dir, model, tkr,sigma=3):

    #image fold
    image_folder = os.path.join(seq_dir, "origin")
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    files.sort() 
    image_paths = [os.path.join(image_folder, f) for f in files]
    seq_name = os.path.basename(seq_dir)
    # # PET dets
    # pet_origin_dir = "/home/data_SSD/zk/pet_outputs/special_case"
    # pet_dir = os.path.join(pet_origin_dir, seq_name,"txt")  
    
    # Build ReID to extract features
    reid_extractor = ReIDFeatureExtractor(batch_size=16, crop_size=10)
    fused_pos_dir = os.path.join(seq_dir,"noNMS","fused_txt_nonms")
    
    res = []
    traj_id = 0
    trajectories = {}  

    # Frame1 - ----------------------------------- 
    for frame in range(0, len(image_paths)):  
        image_name = os.path.basename(image_paths[frame])
        image_prefix = os.path.splitext(image_name)[0]
        
        img_coord_path = os.path.join(fused_pos_dir, f"{image_prefix}_fused_positions.txt")
        if not os.path.exists(img_coord_path):
            print(f"Warning: File {img_coord_path} does not exist, skipping...")
            continue  

        query_data =  reid_extractor.extract_features_from_coordinates(image_paths[frame],img_coord_path)

        if len(query_data) == 0:
            print(f"⚠️ Frame {frame} has no coordinates, skipping processing.")
            continue
        global actual_total_frames
        actual_total_frames += 1 
        seq_id = image_prefix[-3:]
        print(f"Processing: {img_coord_path}, Sequence ID: {seq_id}")

        if frame != len(image_paths) - 1:
            print(f'-------{frame + 1} - {frame + 2}-------')
            # association ------------------------------------------------------------
            mpm = tkr.inferenceMPM([image_paths[frame], image_paths[frame + 1]], model)
            mag = gaussian_filter(np.sqrt(np.sum(np.square(mpm), axis=-1)), sigma=sigma, mode='constant')
            mpm_pos = tkr.getMPMpoints(mpm, mag)

            # initialize trajectory for frame1 ------------------------------------------
            if frame == 0:
                for i in range(len(query_data["coordinate"])):
                  
                    coord = query_data["coordinate"][i]  
                    feat = query_data["feature"][i]
                    
                    traj = Trajectory(
                        obj_id=traj_id,
                        feature=feat,
                        coordinate=coord
                    )
                    trajectories[traj_id] = traj
                  
                    sub_res = [1, traj_id, round(coord[0] - 10,2), round(coord[1] - 10,2), 20, 20,
                               1, 1, -1, -1]
                    traj_id += 1 
                    res.append(sub_res)
                # predict coordinate
                for traj in trajectories.values():
                    traj.predict(mpm)
                continue
  
        # Frame2 - -----------------------------------------------------------------
        else:
            print('------- last_num ---> last_num - 1-------')
            last_num = len(image_paths) - 1
            last_mpm = tkr.inferenceMPM([image_paths[last_num], image_paths[last_num - 1]], model)
            last_mag = gaussian_filter(np.sqrt(np.sum(np.square(last_mpm), axis=-1)), sigma=sigma, mode='constant')
            mpm_pos = tkr.getMPMpoints(last_mpm, last_mag)

            image_name = os.path.basename(image_paths[last_num])
            image_prefix = os.path.splitext(image_name)[0]
            mag = last_mag 
            mpm = last_mpm

        matched_second,unmatched_trks, better_unmatched_dets = \
            assign_cascade(trks=trajectories, dets=query_data, tkr=tkr, mag=mag,image_prefix =image_prefix,mpm = mpm)
        #  successful matched part
        for m in matched_second:
            trajectories[m[1]].update(query_data['feature'][m[0]], query_data['coordinate'][m[0]])
            sub_res = [int(image_prefix[-3:]), m[1],
                        round(query_data['coordinate'][m[0]][0] - 10,2), round(query_data['coordinate'][m[0]][1] - 10,2),
                       20, 20, 1, 1, -1, -1]
            res.append(sub_res)

        for m in unmatched_trks:
            trajectories[m].update(None)

        #  filtered by the Cadidate pool unmatched_dets,
        for traj in better_unmatched_dets.values():
            first = True
            for record in traj.history:
                if first:
                    new_traj_id = traj_id
                    trajectories[new_traj_id] = Trajectory(new_traj_id, record['feature'], record['coord'])
                    first = False
                    traj_id += 1
                else:
                    trajectories[new_traj_id].update(record['feature'], record['coord'])

                sub_res = [record['frame'], new_traj_id,
                        round(record['coord'][0] - 10, 2), round(record['coord'][1] - 10, 2),
                        20, 20, 1, 1, -1, -1]
                res.append(sub_res)

        for traj in trajectories.values():
            traj.predict(mpm)
    #track_result
    seq_head = image_prefix[-6:-3]
    file_path =  os.path.join(save_dir, f"00{seq_head}_diffusion_clip_dis_integrated.txt")
    with open(file_path, "w+") as f:
        for item in res:
            f.write(','.join(map(str, item)) + '\n')

    reset_globals()

    reid_extractor.clear_cache()

def apperance_associate(det_one, trks, detection_indices, track_indices, motion_scores):
    
    scores = get_appreance_scores(det_one,trks,motion_scores)
    cost_matrix = - scores
    pre_matches = []
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
   
    for row, col in zip(row_ind, col_ind):
        
        detection_idx = detection_indices[row]
        track_idx = track_indices[col]
        if scores[row, col] > 0.75:
            pre_matches.append((detection_idx, track_idx, scores[row, col]))

    pre_matches = np.array(pre_matches)

    if pre_matches.shape[0]:
        unmatched_detections = list(set(detection_indices) - set(list(pre_matches[:, 0])))
        unmatched_tracks = list(set(track_indices) - set(list(pre_matches[:, 1])))
      
    else:
        unmatched_detections = detection_indices
        unmatched_tracks = track_indices

    return (pre_matches, unmatched_detections, unmatched_tracks), abs(scores.T)

def level_match(track_indices, detection_one_indices,trks, dets_one, depth, tkr,mag):
    # if track_indices is None:
    #     track_indices = np.arange(len(trks))
    if detection_one_indices is None:
        detection_one_indices = np.arange(len(dets_one))

    unmatched_trks_one = track_indices
    unmatched_dets_one = detection_one_indices
    
    print("---------------第二次匹配----------------")
    matched_second, unmatched_trks_second, unmatched_dets_second = second_associate(trks, dets_one, unmatched_trks_one,
                                                                                    unmatched_dets_one, tkr, mag)

    unmatched_trks_2 = unmatched_trks_second
    unmatched_dets_2 = unmatched_dets_second
  
    return  matched_second,unmatched_trks_2, unmatched_dets_2

# assign_cascade   
def assign_cascade(trks, dets, tkr, mag,image_prefix,mpm):
    track_indices = list(trks.keys())
    detection_one_indices = list(range(len(dets["id"])))
    
    unmatched_dets_one = detection_one_indices
    dets_one = dets
    unmatched_trks_l = []

    for depth in range(1):

        if len(unmatched_dets_one) == 0 :
            break
     
        matched_second,unmatched_trks, unmatched_dets = \
            level_match(track_indices, detection_one_indices, trks, dets_one,
                        depth, tkr, mag)

        global Candidate_pool, can_traj
        frame_id = int(image_prefix[-3:])
        # unmatched_dets_dict  =  dets_one[unmatched_dets]
        better_unmatched_dets = {}
        can_matched = []
        unmatched_trks_can = []
        unmatched_dets_can = []
        if(len(unmatched_dets)!=0)
            if can_traj == 0:  
                for i in unmatched_dets:
                    Candidate_pool[can_traj] = Candidate_traj(can_traj, dets['feature'][i],frame_id,dets['coordinate'][i])  #这里记录特征，坐标，和帧数
                    Candidate_pool[can_traj].predict(mpm) 
                    can_traj += 1
            else:
                
                can_matched,unmatched_trks_can,unmatched_dets_can = candidate_associate(Candidate_pool,dets_one,unmatched_dets,tkr,mag)   #暂定这样
                
                for m in can_matched: 
                    Candidate_pool[m[1]].add_record(frame_id,dets['feature'][m[0]], dets['coordinate'][m[0]])
                    Candidate_pool[m[1]].predict(mpm)
                for m in unmatched_trks_can:
                    Candidate_pool[m].age += 1

                for m in unmatched_dets_can: 
                
                    Candidate_pool[can_traj] = Candidate_traj(can_traj, dets['feature'][m],
                                                            frame_id,dets['coordinate'][m])  
                    Candidate_pool[can_traj].predict(mpm) 
                    can_traj += 1
                 
        # get better_traj    
        better_unmatched_dets = {i: traj for i, traj in enumerate(Candidate_pool.values()) if traj.hits >= 16}
        print("Better unmatched:", len(better_unmatched_dets))

        clean_candidate_pool(Candidate_pool, frame_id)                       

    return matched_second,unmatched_trks,better_unmatched_dets

def fuse_appearance_motion_scores(appearance_scores, motion_scores, modified_scores,distance_threshold=15,max_distance=15.0):
    
    distance_est = distance_threshold * (1 - motion_scores)
    
    alpha = 0.5 - ((distance_est / max_distance) * 0.2)
    appearance_weights = np.clip(alpha, 0.3, 0.5)
    motion_weights = 1.0 - appearance_weights 

    fused_scores = np.multiply(appearance_weights, appearance_scores) + np.multiply(motion_weights, modified_scores)

    min_motion_score = 0.3
    fused_scores[motion_scores < min_motion_score] = 0.0

    return fused_scores

def second_associate(trks, dets_one, unmatched_trks, unmatched_dets, tkr, mag):
   
    unmatched_trks_dict = {tid: trks[tid] for tid in unmatched_trks}
    pred_pos = [traj.predicted_coordinate for traj in unmatched_trks_dict.values()]
    pos = dets_one["coordinate"]
    
    motion_scores = tkr.get_motion_scores(pred_pos, pos, mag, distance_threshold=15)
    
    modified_scores = np.sqrt(motion_scores)
    appearance_scores = get_appreance_scores(dets_one,unmatched_trks_dict, modified_scores)
    
    fused_scores = fuse_appearance_motion_scores(appearance_scores, motion_scores, modified_scores,distance_threshold=15,max_distance=15)
    row_indices, col_indices = linear_sum_assignment(-fused_scores)  

    matched_pairs = []
    unmatched_det_set = set(unmatched_dets)

    for det_idx, trk_idx in zip(row_indices, col_indices):
        score = fused_scores[det_idx, trk_idx]
        if score > 0.65 and det_idx in unmatched_det_set: 
            matched_pairs.append((det_idx, unmatched_trks[trk_idx], score))

    print(f'# Mainly match the associated people: {len(matched_pairs)}')


    matched_det_indices = [m[0] for m in matched_pairs]
    matched_trk_indices = [m[1] for m in matched_pairs]
  
    unmatched_trks_2 = np.setdiff1d(unmatched_trks, matched_trk_indices)
    unmatched_dets_2 = np.setdiff1d(unmatched_dets, matched_det_indices)
    print(f'# unmatched_trk: {len(unmatched_trks_2)}')
    print(f'# unmatched_det: {len(unmatched_dets_2)}')

    return matched_pairs, unmatched_trks_2, unmatched_dets_2

def candidate_associate(candidate_pool, dets_one,unmatched_dets, tkr, mag):

    candidate_keys = list(candidate_pool.keys())
    pred_pos = [traj.predicted_coordinate for traj in candidate_pool.values()] 
    unmatched_dets_dict = dets_one[unmatched_dets]
    pos = unmatched_dets_dict["coordinate"]
    
    motion_scores = tkr.get_motion_scores(pred_pos, pos, mag, distance_threshold=6)
    modified_scores = np.sqrt(motion_scores) 
    appearance_scores = get_appreance_scores(unmatched_dets_dict, candidate_pool, modified_scores,weight = 0.4,distance_threshold=6)
    row_indices, col_indices = linear_sum_assignment(-appearance_scores)  
    
    matched_pairs = []
    for det_idx, trk_idx in zip(row_indices, col_indices):
        score = appearance_scores[det_idx, trk_idx]
        if score > 0.8:  
            matched_pairs.append((unmatched_dets[det_idx], candidate_keys[trk_idx], score))

    print(f'#candidate pool associated people: {len(matched_pairs)}')


    matched_det_indices = [m[0] for m in matched_pairs]
    matched_candidate_indices = [m[1] for m in matched_pairs]

    unmatched_trks_cand = np.setdiff1d(candidate_keys, matched_candidate_indices)
    unmatched_dets_cand = np.setdiff1d(unmatched_dets, matched_det_indices)
    print(f'# unmatched_trk in candidate_pool: {len(unmatched_trks_cand)}')
    print(f'# unmatched_det: {len(unmatched_dets_cand)}')

    return matched_pairs, unmatched_trks_cand, unmatched_dets_cand

def get_appreance_scores(dets_one,trks, motion_scores,weight = 0.6,distance_threshold=15):
    
    gallery_ids = list(trks.keys())
    gallery_features = np.array([trks[tid].get_weighted_feature() for tid in gallery_ids])
    query_features = dets_one["feature"]
    n_query = len(query_features)
    trun_size = max(n_query,len(gallery_ids))
    if trun_size > 100:
        args.truncation_size = (trun_size // 100) * 100
    elif trun_size >= 10:  # 10 ≤ trun_size ≤ 100
        args.truncation_size = (trun_size // 10) * 10
    else:  # trun_size < 10
        args.truncation_size = trun_size  
    diffusion = Diffusion(np.vstack([query_features, gallery_features]), args.cache_dir)
    offline = diffusion.get_offline_results(args.truncation_size, args.kd)
    features = preprocessing.normalize(offline, norm="l2", axis=1)
    print(f"offline.shape: {offline.shape}")
    scores = features[:n_query] @ features[n_query:].T

    # normalized_queries = preprocessing.normalize(det_one["feature"], norm="l2", axis=1)
    # normalized_gallery = preprocessing.normalize(gallery_features, norm="l2", axis=1)
    # scores = normalized_queries @ normalized_gallery.T    # normal compute
    print("appearance_scores----------------------")
    query_coordinates = np.array(dets_one["coordinate"])  
    print("the num of dets", query_coordinates.shape)
    gallery_coordinates = np.array([trks[tid].current_coordinate for tid in gallery_ids])
    print("the num of tracks", gallery_coordinates.shape)
    distances = cdist(query_coordinates, gallery_coordinates, metric='euclidean')
    scores[distances > distance_threshold] = float('0')

    fused_scores = scores * weight + motion_scores * (1 - weight)

    return fused_scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',
                        type=str,
                        default='./cache',
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--query_path',
                        type=str,
                        required=False,
                        help="""
                        Path to query features
                        """)
    parser.add_argument('--gallery_path',
                        type=str,
                        required=False,
                        help="""
                        Path to gallery features
                        """)
    parser.add_argument('--gnd_path',
                        type=str,
                        help="""
                        Path to ground-truth
                        """)
    parser.add_argument('-n', '--truncation_size',
                        type=int,
                        default=300,
                        help="""
                        Number of images in the truncated gallery
                        """)
    args = parser.parse_args()
    args.kq, args.kd = 10, 50
    return args

if __name__ == '__main__':
    
    start_time = time.time()  
    args = parse_args()
    model_path='DroneCrowd_weight/model_best.pth'
    tkr = trackp(mag_th=0.3,itp=5,sigma=3,maxv=255,image_size=(1080, 1920))  

    model = tkr.loadModel(model_path)
    over_seq = ["00070"]
    root_dir = "/home/data_SSD/zk/dataset/test_data"
   
    sequence_dirs = sorted([
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])
    output_dir = "/home/data_SSD/zk/result_test/integrated_result_test"
    os.makedirs(output_dir, exist_ok=True)
    for seq_dir in sequence_dirs:
        seq_name = os.path.basename(seq_dir)
        # if seq_name  not in over_seq:
        #     continue
        print(f"\n==> Start processing sequence: {seq_name}")
        mpm_track(seq_dir = seq_dir,save_dir= output_dir, model = model,tkr = tkr,sigma = 3)
        torch.cuda.empty_cache()  
    
    end_time = time.time() 
    expected_total_frames = 9000
    elapsed_time = end_time - start_time
    if elapsed_time > 0:
        fps = actual_total_frames / elapsed_time
    print(f"\n A total of {total_sequences} sequences were processed.")
    print(f" Actual frame processing count: {total_frames} (Expected: {expected_total_frames})")
    print(f" Total elapsed time: {elapsed_time:.2f} seconds, processing speed - average FPS: {fps:.2f}")

    