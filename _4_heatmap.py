from pathlib import Path
import os
import cv2
import numpy as np
# load videopy editor
import matplotlib.pyplot as plt
from moviepy import editor as mpy

# local imports
from heatmap import (
    vis_heatmap,
    make_heatmap,
    plot_heatmap,
    from_txt_to_det,
    from_det_to_cir,
    cir_to_heatmap,)

# functions

def load_frame(dir):
    file_mp4 = [m for m in dir.parent.iterdir()\
        if ".mp4" in m.suffix][0]
    with mpy.VideoFileClip(str(file_mp4)) as video:
        # get frame of the first second
        frame = video.get_frame(1)
    return frame

def check_boundary(frame, point):
    plt.imshow(frame)
    plt.scatter(*zip(*point), 
                c="r", s=30, marker="x")


dir_beef = [f / "labels" \
    for f in (Path.cwd() / "logs").iterdir() \
    if "beef" in f.name]
dir_beef.sort()


points = dict(
    square = [
        (0, 0),
        (0, 1000),
        (1000, 1000),
        (1000, 0),
    ],
    beef_1 = [
        (1110, 80), # top left
        (250, 690), # bottom left
        (1100, 1530), # bottom right
        (1550, 150), # top right
    ],
    beef_2 = [
        (300, 70), # top left
        (750, 1500), # bottom left
        (1650, 630), # bottom right
        (800, 0), # top right
    ],
    beef_3 = [
        (800, -20), # top left
        (130, 850), # bottom left
        (1380, 1580), # bottom right
        (1260, -10), # top right
    ],
    beef_4 = [
        (400, 150), # top left
        (900, 1600), # bottom left
        (1800, 550), # bottom right
        (950, -10), # top right
    ],
)


i = 2
d = dir_beef[i]
point = points[f"beef_{i + 1}"]

frame = load_frame(d)
check_boundary(frame, point)




VIDEO_W, VIDEO_H = frame.shape[1], frame.shape[0]
# find holomatrix to turn from beef_2 to square
matrixH = cv2.findHomography(
    np.array(points[f"beef_{i + 1}"], dtype=np.float32),
    np.array(points["square"], dtype=np.float32),
)[0]

R = 0.015
# from 8:30 to 16:30 -> 8 hours
# 1/4 = 2 hours
ls_txt = [f for f in d.iterdir()]
ls_txt.sort(key=os.path.getmtime)
n_det = len(ls_txt)
ls_txt = ls_txt[:n_det//8]

ls_det = from_txt_to_det(ls_txt)
ls_cir = from_det_to_cir(ls_det, r=R)
# apply holomatrix to beef_2 (x, y, r)
np_cir = np.array(ls_cir)
np_cir = np_cir[:, :2]
np_cir[:, 0] = VIDEO_W * np_cir[:, 0]
np_cir[:, 1] = VIDEO_H * np_cir[:, 1]
np_cir = np_cir.reshape(-1, 1, 2)
new_point = cv2.perspectiveTransform(np_cir, matrixH)
new_point = new_point.reshape(-1, 2)
# plot new points
plt.scatter(*zip(*new_point), c="b", s=30, marker="x")

# expand the radius
new_point = np.concatenate([new_point, 
                            R * np.ones((new_point.shape[0], 1))], axis=1)

new_point = new_point[new_point[:, 0] >= 0]
new_point[:, 0] /= VIDEO_W
new_point[:, 1] /= VIDEO_H

heatmap = cir_to_heatmap(new_point)
plot_heatmap(heatmap, max_value=5)
