import os
import numpy as np
import scipy.io as sio
from numba import njit
from multiprocessing import Pool

def load_groundtruth(list_path, gt_path, num_imgs=500, default_iou_thr=25):
    with open(list_path, 'r') as f:
        all_sequences = [line.strip() for line in f if line.strip()]
    num_vids = len(all_sequences)

    gt_track_img_ids = []
    gt_track_labels = []
    gt_track_bboxes = []
    gt_track_thr = []

    for seq_id in all_sequences:
        sequence_name = seq_id  # e.g. "00001"
        gt_mat_file = os.path.join(gt_path, f"{sequence_name}.mat")
       
        clean_txt_file = os.path.join(gt_path, f"{sequence_name}_clean.txt")
        if not os.path.exists(clean_txt_file):
            if not os.path.exists(gt_mat_file):
                raise FileNotFoundError(f"GT file {gt_mat_file} not found.")
            data = sio.loadmat(gt_mat_file)
            anno = data['anno']  # shape (N, 4) frame, id, x, y (or similar)
            rec = np.hstack([
                anno[:, [0]] + 1,               # frame idx +1
                anno[:, 1:4],                   # track id + bbox (id,x,y)
                np.tile([20, 20, 1, 1, 0, 0], (anno.shape[0], 1))
            ])
            np.savetxt(clean_txt_file, rec, fmt='%.6f', delimiter=',')
        else:
            rec = np.loadtxt(clean_txt_file, delimiter=',')

        tracks = []
        recs = [None] * num_imgs
        num_tracks = 0

        for i in range(1, num_imgs + 1):
            idx = rec[:, 0] == i
            currec = rec[idx, :]
            recs[i - 1] = currec
            for j in range(currec.shape[0]):
                trackid = int(currec[j, 1])
                if trackid not in tracks:
                    tracks.append(trackid)
                    num_tracks += 1

        gt_img_ids_per_track = [[] for _ in range(num_tracks)]
        gt_labels_per_track = [-1] * num_tracks
        gt_bboxes_per_track = [[] for _ in range(num_tracks)]
        gt_thr_per_track = [[] for _ in range(num_tracks)]

        for i in range(num_imgs):
            currec = recs[i]
            for j in range(currec.shape[0]):
                trackid = int(currec[j, 1])
                k = tracks.index(trackid)
                gt_img_ids_per_track[k].append(i + 1)
                if gt_labels_per_track[k] == -1:
                    gt_labels_per_track[k] = 1  
                # bbox  [x1, y1, x2, y2]
                x1 = currec[j, 2]
                y1 = currec[j, 3]
                x2 = x1 + currec[j, 4] - 1
                y2 = y1 + currec[j, 5] - 1
                gt_bboxes_per_track[k].append([x1, y1, x2, y2])
                gt_thr_per_track[k].append(default_iou_thr)

        for k in range(num_tracks):
            gt_img_ids_per_track[k] = np.array(gt_img_ids_per_track[k], dtype=int)
            gt_bboxes_per_track[k] = np.array(gt_bboxes_per_track[k], dtype=np.float32)
            gt_thr_per_track[k] = np.array(gt_thr_per_track[k], dtype=np.float32)

        gt_track_img_ids.append(gt_img_ids_per_track)
        gt_track_labels.append(np.array(gt_labels_per_track, dtype=int))
        gt_track_bboxes.append(gt_bboxes_per_track)
        gt_track_thr.append(gt_thr_per_track)

    return gt_track_img_ids, gt_track_labels, gt_track_bboxes, gt_track_thr, num_vids


def load_tracker_results(list_path, res_path, trackName):
    # read seq_name 
    with open(list_path, 'r') as f:
        all_sequences = [line.strip() for line in f if line.strip()]
    
    track_confs = []
    track_img_ids = []
    track_labels = []
    track_bboxes = []

    for seq_id in all_sequences:
        sequence_name = seq_id  
        track_file = os.path.join(res_path, f"{sequence_name}_{trackName}.txt")
        if not os.path.exists(track_file):
            raise FileNotFoundError(f"Tracker file {track_file} not found.")
        
        res = np.loadtxt(track_file, delimiter=',')
        if res.ndim == 1:
            res = res[np.newaxis, :] 

        img_ids_seq = res[:, 0].astype(int)      # frame ID
        track_ids_seq = res[:, 1].astype(int)    # track ID
        bboxes_xywh = res[:, 2:6]                # (x, y, w, h)
        confs_seq = res[:, 6]                    # values
        
        if res.shape[1] > 7:
            labels_seq = res[:, 7].astype(int)
        else:
            labels_seq = np.ones_like(img_ids_seq, dtype=int)

        bboxes_xyxy = np.zeros_like(bboxes_xywh)
        bboxes_xyxy[:, 0] = bboxes_xywh[:, 0]                # x1
        bboxes_xyxy[:, 1] = bboxes_xywh[:, 1]                # y1
        bboxes_xyxy[:, 2] = bboxes_xywh[:, 0] + bboxes_xywh[:, 2]-1  # x2 = x + w
        bboxes_xyxy[:, 3] = bboxes_xywh[:, 1] + bboxes_xywh[:, 3]-1  # y2 = y + h

        unique_tracks = np.unique(track_ids_seq)
        
        confs_per_track = []
        img_ids_per_track = []
        labels_per_track = []
        bboxes_per_track = []

        for tid in unique_tracks:
            idxs = np.where(track_ids_seq == tid)[0]
            confs_per_track.append(np.mean(confs_seq[idxs]))         
            img_ids_per_track.append(img_ids_seq[idxs])
            labels_per_track.append(labels_seq[idxs][0])             
            bboxes_per_track.append(bboxes_xyxy[idxs].T)           

        track_confs.append(np.array(confs_per_track))
        track_img_ids.append(img_ids_per_track)
        track_labels.append(np.array(labels_per_track))
        track_bboxes.append(bboxes_per_track)

    return track_confs, track_img_ids, track_labels, track_bboxes

# -------------------------------------------------
#  TP / FP 
# -------------------------------------------------
def compute_tp_fp_parallel(
    track_confs, track_img_ids, track_labels, track_bboxes,
    gt_track_img_ids, gt_track_labels, gt_track_bboxes, gt_track_thr,
    defaultTrackThr,num_workers=4
):
    num_vids = len(track_labels)
    args_list = []
    for v in range(num_vids):
        print("Process the num of sequence is ------->>>",v+1)
        args_list.append((
            v,track_confs[v], track_img_ids[v], track_labels[v], track_bboxes[v],
            gt_track_img_ids[v], gt_track_labels[v], gt_track_bboxes[v], gt_track_thr[v],
            defaultTrackThr
        ))

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(process_single_video, args_list)

    tp_cell = [res[0] for res in results]
    fp_cell = [res[1] for res in results]

    return tp_cell, fp_cell

def process_single_video(
    v, track_confs_v, track_img_ids_v, track_labels_v, track_bboxes_v,
    gt_track_img_ids_v, gt_track_labels_v, gt_track_bboxes_v, gt_track_thr_v,
    defaultTrackThr
):
    confs_v = np.array(track_confs_v)
    ind = np.argsort(-confs_v, kind="stable")
    confs_v = confs_v[ind]
    track_img_ids_v = [track_img_ids_v[i] for i in ind]
    track_labels_v = [track_labels_v[i] for i in ind]
    track_bboxes_v = [track_bboxes_v[i] for i in ind]

    num_tracks = len(track_labels_v)
    num_gt_tracks = len(gt_track_labels_v)
    num_track_thr = len(defaultTrackThr)

    tp = [np.zeros(num_tracks, dtype=np.uint8) for _ in range(num_track_thr)]
    fp = [np.zeros(num_tracks, dtype=np.uint8) for _ in range(num_track_thr)]
    gt_detected = [np.zeros(num_gt_tracks, dtype=np.uint8) for _ in range(num_track_thr)]

    for m in range(num_tracks):
        img_ids = np.array(track_img_ids_v[m], dtype=np.int64)
        bboxes = np.array(track_bboxes_v[m], dtype=np.float64)
        label = track_labels_v[m]
        # print("the track is",m,"--->num",len(img_ids))
        
        ovmax = np.ones(num_track_thr) * -np.inf
        kmax = np.ones(num_track_thr, dtype=int) * -1

        for n in range(num_gt_tracks):
            gt_label = gt_track_labels_v[n]
            if label != gt_label:
                continue

            gt_img_ids_n = np.array(gt_track_img_ids_v[n], dtype=np.int64)
            gt_bboxes_n = np.array(gt_track_bboxes_v[n], dtype=np.float64)
            gt_thr_n = np.array(gt_track_thr_v[n], dtype=np.float64)

            num_total = len(set(img_ids) | set(gt_img_ids_n))

            ov = compute_ov_numba(img_ids, bboxes, gt_img_ids_n, gt_bboxes_n, gt_thr_n, num_total)

            for o in range(num_track_thr):
                if gt_detected[o][n]:
                    continue
                if ov >= defaultTrackThr[o] and ov > ovmax[o]:
                    ovmax[o] = ov
                    kmax[o] = n

        for o in range(num_track_thr):
            if kmax[o] >= 0:
                tp[o][m] = 1
                gt_detected[o][kmax[o]] = 1
            else:
                fp[o][m] = 1

    return tp, fp
@njit
def compute_ov_numba(
    img_ids, bboxes, gt_img_ids_n, gt_bboxes_n, gt_thr_n, num_total
):

    num_obj = img_ids.shape[0]
    num_gt_obj = gt_img_ids_n.shape[0]
    num_matched = 0

    for j in range(num_obj):
        id_j = img_ids[j]

        k = -1
        for idx in range(num_gt_obj):
            if gt_img_ids_n[idx] == id_j:
                k = idx
                break
        if k == -1:
            continue

        bb = bboxes[:, j]
        bbgt = gt_bboxes_n[k]

        det_center_x = 0.5 * (bb[0] + bb[2])
        det_center_y = 0.5 * (bb[1] + bb[3])
        gt_center_x = 0.5 * (bbgt[0] + bbgt[2])
        gt_center_y = 0.5 * (bbgt[1] + bbgt[3])

        dist = ((det_center_x - gt_center_x) ** 2 + (det_center_y - gt_center_y) ** 2) ** 0.5

        if dist <= gt_thr_n[k]:
            num_matched += 1

    if num_total == 0:
        return 0.0
    else:
        ov = num_matched / num_total
        return ov
# -------------------------------------------------
# AP 
# -------------------------------------------------
def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap

def calc_ap(track_confs, tp_cell, fp_cell, num_track_per_class, defaultTrackThr):
    num_thr = len(defaultTrackThr)
    aps, recalls, precisions = [], [], []

    confs_all = np.concatenate(track_confs)
    sort_ind = np.argsort(-confs_all)

    for o in range(num_thr):
        tp_all, fp_all = [], []
        for v in range(len(tp_cell)):
            tp_all.append(tp_cell[v][o])
            fp_all.append(fp_cell[v][o])

        tp_all = np.concatenate(tp_all)[sort_ind]
        fp_all = np.concatenate(fp_all)[sort_ind]

        tp_cum = np.cumsum(tp_all)
        fp_cum = np.cumsum(fp_all)

        rec = tp_cum / float(num_track_per_class)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
        
        ap = voc_ap(rec, prec) * 100
        aps.append(ap)
        recalls.append(rec)
        precisions.append(prec)

    mean_ap = np.mean(aps)
    return aps, recalls, precisions, mean_ap

def calc_ap_with_log(res_path, trackName,track_confs, tp_cell, fp_cell, num_track_per_class, defaultTrackThr, log_path="ap_log.txt"):
    num_thr = len(defaultTrackThr)
    aps, recalls, precisions = [], [], []

    confs_all = np.concatenate(track_confs)
    sort_ind = np.argsort(-confs_all, kind='stable')

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-----------------------------------------------------\n")
        f.write("------computing AP------\n")

        for o in range(num_thr):
            tp_all, fp_all = [], []
            for v in range(len(tp_cell)):
                tp_all.append(tp_cell[v][o])
                fp_all.append(fp_cell[v][o])

            tp_all = np.concatenate(tp_all)[sort_ind]
            fp_all = np.concatenate(fp_all)[sort_ind]

            tp_cum = np.cumsum(tp_all)
            fp_cum = np.cumsum(fp_all)

            rec = tp_cum / float(num_track_per_class)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

            ap = voc_ap(rec, prec) * 100
            aps.append(ap)
            recalls.append(rec)
            precisions.append(prec)

        mean_ap = np.mean(aps)
        f.write(f"---{res_path}-->>{trackName}---1-8_RegenTrack_25_5\n")
        f.write(f"Mean AP:\t\t {mean_ap:.2f}%\n")
        f.write(" = = = = = = = = \n")
        for t in range(num_thr):
            f.write(f"Mean AP@{defaultTrackThr[t]:.2f}:\t {aps[t]:.2f}%\n")
        f.write(" = = = = = = = = \n")

    return aps, recalls, precisions, mean_ap

if __name__ == "__main__":

    list_path = "test_list/test_dronebird.txt"# list of seq
    gt_path = "/tracking/DroneBird/annotations" #ground_truth
    
    
    res_path = "./result_motion_25_5"
    trackName = "dronebird" 
    log_path = "dronebird/ap_log.txt"
    
    default_track_thr = [0.10, 0.15, 0.20]  

    print("load Ground Truth...")
    gt_track_img_ids, gt_track_labels, gt_track_bboxes, gt_track_thr, num_vids = load_groundtruth(list_path, gt_path)

    print("load Tracker results...")
    track_confs, track_img_ids, track_labels, track_bboxes = load_tracker_results(list_path, res_path,trackName)

    print("（num_track_per_class）...")
    num_track_per_class = sum(len(v) for v in gt_track_labels)  

    print(f"the num of tracks ：{num_track_per_class}")

    print(" TP / FP ...")
    tp_cell, fp_cell = compute_tp_fp_parallel(
        track_confs, track_img_ids, track_labels, track_bboxes,
        gt_track_img_ids, gt_track_labels, gt_track_bboxes, gt_track_thr,
        default_track_thr,num_workers = 4
    )

    print(" AP ...")
    aps, recalls, precisions,mean_ap = calc_ap_with_log(
        res_path,trackName, track_confs, tp_cell, fp_cell, num_track_per_class, default_track_thr,log_path
    )
