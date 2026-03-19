import cv2
import numpy as np
import os
from collections import defaultdict


track_file = '00069_clean.txt'
# track_file = '00105_diffusion_clip_dis_isactive.txt'
image_dir = 'test_data/00069/origin'
output_dir = 'result_visualization/traj_clean_frames_069'
os.makedirs(output_dir, exist_ok=True)

# tracking track_id -> list of (frame_id, x, y)
trajectories = defaultdict(list)
id_colors = dict()

with open(track_file) as f:
    for line in f:
        items = line.strip().split(',')
        if len(items) < 6:
            continue
        frame_id = int(items[0])
        track_id = int(items[1])
        x = float(items[2])
        y = float(items[3])
        w = float(items[4])
        h = float(items[5])
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        trajectories[track_id].append((frame_id, cx, cy))

#frame_id -> list of (track_id, x, y)
frames = defaultdict(list)
for track_id, points in trajectories.items():
    for frame_id, x, y in points:
        frames[frame_id].append((track_id, x, y))


for track_id in trajectories:
    id_colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
    
id_history = defaultdict(list)

for frame_id in range(1, 301):
    img_path = os.path.join(image_dir, f'img069{frame_id:03d}.jpg')
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Warning: Frame {frame_id} not found.")
        continue

    for track_id, x, y in frames.get(frame_id, []):
        color = id_colors[track_id]

        id_history[track_id].append((x, y))

        if len(id_history[track_id]) >= 2:
            for i in range(1, len(id_history[track_id])):
                pt1 = id_history[track_id][i - 1]
                pt2 = id_history[track_id][i]
                cv2.line(frame, pt1, pt2, color, thickness=2)

        cv2.circle(frame, (x, y), 4, color, thickness=-1)
        cv2.putText(frame, str(track_id), (x + 5, y - 5), 
             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, lineType=cv2.LINE_AA)

    out_path = os.path.join(output_dir, f'output_{frame_id:06d}.jpg')
    cv2.imwrite(out_path, frame)

print("✅ Done: 300frame images have been saved。")
