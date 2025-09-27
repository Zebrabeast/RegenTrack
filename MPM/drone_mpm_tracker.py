import os
import colorsys
import random
from glob import glob

import cv2
import numpy as np
from mpm_track_parts import trackp
from scipy.ndimage import gaussian_filter

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def SaveCoor(save_dir, image_names, coor_list):
    f = h5py.File(os.path.join(save_dir, image_names).replace('.jpg','.h5'), 'w+')
    f['coor_list'] = coor_list
    f.close()

    
def SaveAsso(save_dir, id, ass_list):
    f = h5py.File(os.path.join(save_dir,Ass+str(id)+'.h5'), 'w+')
    f['asso_list'] = ass_list
    f.close()


def ShowPoints(image_name, coor_list):
    root_path = '/showresult/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    img = cv2.imread(image_name)
    for pid in range(len(coor_list)):
        cv2.circle(img, (int(coor_list[pid][0]), int(coor_list[pid][1])), 2, (0, 0, 255), 2)
    result_name = os.path.basename(image_name).split('.')[0]+'_3'+'.jpg'
    save_path = os.path.join(root_path, result_name)
    cv2.imwrite(save_path, img)


def track(image_names, model_path, save_dir, mag_th=0.3, itp=5, sigma=3, maxv=255):
    seq_id1 = os.path.basename(image_names[0])[3:6]
    print(seq_id1)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir1 = 'MPM/Asso'
    tkr = trackp(mag_th, itp, sigma, maxv, (1080, 1920))  # tracker

    model = tkr.loadModel(model_path)

    ID_COLOR = [colorsys.hsv_to_rgb(h, 0.8, 0.8) for h in np.linspace(0, 1, 100)]
    random.shuffle(ID_COLOR)
    ID_COLOR = np.array(ID_COLOR)

    # track_tab: Tracking records. shape is (, 11)
    # 0:frame, 1:id, 2:x, 3:y, 4:warped_x, 5:warped_y, 6:peak_value, 7:warped_pv, 8:climbed_wpv, 9:dis, 10:pid
    track_tab = []
    not_ass_tab = np.empty((0, 11))  # Not association table
    pop_tab = np.empty((0, 11))  # Appear table
    div_tab = np.empty((0, 11))  # Division table

    # Frame0 - farme1
    print('-------0 - 1-------')
    pre_mpm = tkr.inferenceMPM([image_names[0], image_names[1]], model)
    print(pre_mpm.shape)
    pre_mag = gaussian_filter(np.sqrt(np.sum(np.square(pre_mpm), axis=-1)), sigma=sigma, mode='constant')
    print(pre_mag)
    pre_pos = tkr.getMPMpoints(pre_mpm, pre_mag)
    for i, each_pos in enumerate(pre_pos):
        # register frame(0 & 1)
        track_tab.append([0, i + 1] + each_pos[2:4] + [0, 0, 0, 0, 0, 0, 0])
        track_tab.append([1, i + 1] + each_pos + [pre_mag[int(each_pos[1]), int(each_pos[0])], 0, 0, 0])
    track_tab = np.array(track_tab)
    # saveResultImage(1, image_names[0], image_names[1], pre_mag, np.zeros(pre_mag.shape),
    #                 track_tab[track_tab[:, 0] == 1], div_tab)
    print(len(track_tab))
    if len(track_tab) > 0:
        pre_pos = track_tab[track_tab[:, 0] == 1]
    else:
        pre_pos = []
    # pre_pos = track_tab[track_tab[:, 0] == 1]
    new_id = len(pre_pos) + 1
    
    # Frame2 - -------------------------------------------------------------------
    for frame in range(2, len(image_names)):
        print(f'-------{frame - 1} - {frame}-------')
        # association ------------------------------------------------------------
        mpm = tkr.inferenceMPM([image_names[frame - 1], image_names[frame]], model)
        mag = gaussian_filter(np.sqrt(np.sum(np.square(mpm), axis=-1)), sigma=sigma, mode='constant')
        pos = tkr.getMPMpoints(mpm, mag)
        add_ass, add_not_ass, add_div, add_pop, new_id = tkr.associateCells(frame, pos, pre_pos, pre_mag, new_id)
        div_tab = np.append(div_tab, add_div, axis=0)
        if len(pop_tab) > 0:
            pop_tab = np.append(pop_tab, add_pop, axis=0)
        # 这个地方也要加一个判定：
        if len(add_not_ass) > 0:
            not_ass_tab = np.append(not_ass_tab, add_not_ass, axis=0)
        # not_ass_tab = np.append(not_ass_tab, add_not_ass, axis=0)
        # saveResultImage(frame, image_names[frame - 1], image_names[frame], mag, pre_mag, ass_tab, add_div_tab)
        pre_mag = mag.copy()
        pre_pos = add_ass.copy()

        if len(add_ass) > 0:
            if len(track_tab) > 0:
                track_tab = np.append(track_tab, add_ass, axis=0)
            else:
                track_tab = add_ass

        track_results = []
        for tid in range(len(track_tab)):
            track_result = [int(track_tab[tid][0])+1, int(track_tab[tid][1]), int(track_tab[tid][2]), int(track_tab[tid][3]), 20, 20, 1, 1, -1, -1]
            track_results.append(track_result)

        f = open('MPM/track_result20/00{}_MPM.txt'.format(seq_id1), 'w+')
        for j in range(len(track_results)):
            f.write(str(track_results[j][0]))
            for k in range(1, len(track_results[j])):
                f.write(',' + str(track_results[j][k]))
            f.write('\n')
        f.close()

    # tkr.saveCTK(track_tab, not_ass_tab, div_tab, save_dir)
    return
if __name__ == '__main__':
    images_path = '/home/ly/yp/DroneCrowd/test_data/test_data/images'
    images_list = glob(r'/home/ly/yp/DroneCrowd/test_data/test_data/images/*')
    images_list.sort()
    print(images_list[0:300])
    vid_num = len(images_list)/300
    print(vid_num)
    for vid in range(int(vid_num)):
        demo = track(
            image_names=images_list[int(vid*300):int((vid+1)*300)],
            model_path='checkpoints7/model_best.pth',
            save_dir='MPM/Pcoor')
