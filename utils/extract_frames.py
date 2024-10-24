"""
args:
    -i: input video file
    -f: how many frames to extract per second)
    -o: output directory
"""

import os
import numpy as np
import tqdm
import argparse
from moviepy.editor import VideoFileClip
from PIL import Image

# path_video = "/Users/niche/Downloads/Beef-02_20241021_overview.mp4"
# dir_out = "/Users/niche/Downloads/beef"
# fps_tgt = 1
# prefix = "Beef-02"

def main(args):
    path_video = args.i
    dir_out = args.o
    fps_tgt = args.f
    prefix = args.p
    
    os.makedirs(dir_out, exist_ok=True)

    with VideoFileClip(path_video) as videofile:
        dur = videofile.duration # in seconds
        ls_t = np.arange(0, dur, 1 / fps_tgt)
        # rewire to tqdm
        for i, t in enumerate(tqdm.tqdm(ls_t, desc="Extracting frames")):
            img = videofile.get_frame(t)
            img = Image.fromarray(img)
            
            img.save(os.path.join(dir_out, f"{prefix}_{i:02d}.jpg"))
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input video file")
    parser.add_argument("-f", help="frame rate", type=int)
    parser.add_argument("-o", help="output directory")
    parser.add_argument("-p", default="frame", 
                        help="prefix for output files")
    args = parser.parse_args()
    main(args)
