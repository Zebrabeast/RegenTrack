import cv2
import os

def draw_trajectory(data_file, image_path, output_path, draw_box=True, track_id=None):

    image = cv2.imread(image_path)
    if image is None:
        print(f"the path is : {image_path}")
        return

    with open(data_file, 'r') as f:
        lines = f.readlines()

    points = []

    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 6:
            continue
        frame_id = int(parts[0].strip())
        obj_id = int(parts[1].strip())
        x = int(float(parts[2].strip()))
        y = int(float(parts[3].strip()))
        w = int(float(parts[4].strip()))
        h = int(float(parts[5].strip()))
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        if track_id is not None and obj_id != track_id:
            continue

        if draw_box:
            cv2.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 1)
            # cv2.rectangle(image, (int(wx - w / 2), int(wy - h / 2)), (int(wx + w / 2), int(wy + h / 2)), (0, 0, 255), 2)
        else:
            cv2.circle(image, (cx, cy), 1, (0, 0, 255), -1)


    cv2.imwrite(output_path, image)
    print(f"saved {output_path}")

# 示例调用
if __name__ == "__main__":

    data_file = "00074_diffusion_clip_dis_integrated.txt"
    image_path = "/test_data/00074/origin/img074001.jpg"
    # image_path = "valid/mpm/trajectory_orgin_div_test1_2.png"
    output_path = "one_track/trajectory_074_frame_1.png"
    draw_box = False  # True box,False: point
    track_id = None  # special object ID, None for all tracks 

    draw_trajectory(data_file, image_path, output_path, draw_box, track_id)